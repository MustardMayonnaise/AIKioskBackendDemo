import copy
import csv
import json
import logging
import re
import time
from pathlib import Path
from uuid import uuid4


logger = logging.getLogger(__name__)
MAX_CART_LOG_TEXT_CHARS = 800

# 대화에서 추출한 주문 정보를 세션별 장바구니 응답 형식으로 누적한다.
class ChatCartService:

    # 장바구니 템플릿, 메뉴 목록, 가격표를 초기화한다.
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.template = self._load_template(base_dir / "data" / "cart.json")
        self.sessions: dict[str, dict] = {}
        self.menus = self._load_menus(base_dir / "models" / "vectorstore" / "SUBWAY_MENU.csv")
        self.prices = self._load_prices(base_dir / "models" / "vectorstore" / "SUBWAY_MENU.csv")

    # 현재 대화 입력과 RAG 주문 상태를 합쳐 최신 장바구니 응답을 만든다.
    def build_response(
        self,
        session_id: str | None,
        user_text: str,
        answer: str,
        order_info: str | None = None,
        order_process: int | None = None,
    ) -> dict:
        
        # 세션과 기존 장바구니 상태를 준비한다.
        started_at = time.perf_counter()
        session_id = session_id or f"s_{uuid4().hex[:12]}"
        cart = copy.deepcopy(self.sessions.get(session_id, self.template))
        cart["session_id"] = session_id
        cart["answer"] = answer

        # RAG가 반환한 주문 상태를 파싱하고, 현재 누락된 주문 단계를 계산한다.
        active_order = cart["active_order"]
        previous_step = cart.get("current_step") or self._next_step(active_order)
        order_state = self._parse_order_state(order_info or "")
        logger.info(
            "Cart parse started (session_id=%s, previous_step=%s, order_process=%s, user_chars=%d, answer_chars=%d, order_info_chars=%d, order_state=%s, user_text=%s)",
            session_id,
            previous_step,
            order_process,
            len(user_text or ""),
            len(answer or ""),
            len(order_info or ""),
            order_state,
            _compact_log_text(user_text),
        )

        # 메뉴와 사이즈처럼 단일 값으로 결정되는 기본 주문 항목을 반영한다.
        menu = self._find_menu(self._field_sources(order_state, ["메인 메뉴", "메인메뉴", "샌드위치"], user_text, previous_step, {"MENU_SELECT"}))
        if menu is not None:
            active_order["order_item_id"] = active_order["order_item_id"] or f"oi_{uuid4().hex[:12]}"
            active_order["menu"] = menu

        size = self._find_one(self._field_sources(order_state, ["사이즈", "크기"], user_text, previous_step, {"MENU_SELECT", "SIZE_SELECT"}), SIZE_OPTIONS)
        if size is not None:
            active_order["size"] = size["id"]

        # 빵과 치즈는 옵션 참조 형태로 저장한다.
        for field, state_key, options, expected_steps in (
            ("bread", ["빵", "브레드"], BREAD_OPTIONS, {"BREAD_SELECT"}),
            ("cheese", ["치즈"], CHEESE_OPTIONS, {"CHEESE_SELECT"}),
        ):
            value = self._find_one(self._field_sources(order_state, state_key, user_text, previous_step, expected_steps), options)
            if value is not None:
                active_order[field] = self._ref(value)

        # 토스팅 여부와 여러 개를 선택할 수 있는 옵션 그룹을 반영한다.
        toast = self._find_toast(self._field_sources(order_state, ["토스팅", "데우기", "굽기"], user_text, previous_step, {"TOAST_SELECT"}))
        if toast is not None:
            active_order["is_toasted"] = toast

        self._apply_many(
            active_order,
            "vegetables",
            self._field_sources(order_state, ["야채", "채소"], user_text, previous_step, {"VEGETABLE_SELECT"}),
            VEGETABLE_OPTIONS,
        )
        self._apply_many(
            active_order,
            "sauces",
            self._field_sources(order_state, ["소스"], user_text, previous_step, {"SAUCE_SELECT"}),
            SAUCE_OPTIONS,
        )
        self._apply_many(
            active_order,
            "side_menu_items",
            self._field_sources(order_state, ["사이드 및 음료", "사이드/음료", "사이드", "음료"], user_text, previous_step, {"SIDE_SELECT"}),
            SIDE_OPTIONS,
        )
        self._apply_many(
            active_order,
            "extras",
            self._field_sources(order_state, ["추가 재료", "추가재료", "추가"], user_text, previous_step, {"ORDER_CONFIRM"}),
            EXTRA_OPTIONS,
        )

        quantity = self._find_quantity(self._field_sources(order_state, ["수량", "개수"], user_text, previous_step, {"MENU_SELECT"}))
        if quantity is not None:
            active_order["quantity"] = quantity

        # 가격과 다음 주문 단계를 갱신한 뒤 세션 상태를 저장한다.
        self._apply_price(active_order)
        cart["current_step"] = self._next_step(active_order)
        self.sessions[session_id] = copy.deepcopy(cart)
        logger.info(
            "Cart parse completed (session_id=%s, elapsed_ms=%.2f, previous_step=%s, current_step=%s, menu_id=%s, size=%s, bread_id=%s, cheese_id=%s, toasted=%s, vegetables=%s, sauces=%s, sides=%s, extras=%s, total_price=%s)",
            session_id,
            (time.perf_counter() - started_at) * 1000,
            previous_step,
            cart["current_step"],
            active_order["menu"]["id"],
            active_order["size"],
            active_order["bread"]["id"],
            active_order["cheese"]["id"],
            active_order["is_toasted"],
            [item.get("id") for item in active_order.get("vegetables", [])],
            [item.get("id") for item in active_order.get("sauces", [])],
            [item.get("id") for item in active_order.get("side_menu_items", [])],
            [item.get("id") for item in active_order.get("extras", [])],
            active_order["price"]["total_price"],
        )
        return cart
    
    # 저장된 세션 장바구니를 조회하되 응답에서 오디오 필드는 제외한다.
    def get_response(self, session_id: str) -> dict | None:
        cart = self.sessions.get(session_id)
        if cart is None:
            return None
        response = copy.deepcopy(cart)
        response.pop("audio", None)
        return response

    # 기본 장바구니 JSON 템플릿을 로드한다.
    @staticmethod
    def _load_template(path: Path) -> dict:
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    # 메뉴 CSV에서 샌드위치 메뉴명과 별칭을 추출한다.
    @staticmethod
    def _load_menus(path: Path) -> list[dict]:
        if not path.exists():
            return []

        # 중복 메뉴를 제거하면서 긴 메뉴명이 먼저 매칭되도록 정렬 가능한 목록을 만든다.
        seen: set[str] = set()
        menus: list[dict] = []
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                content = row.get("content", "")
                match = re.match(r"(.+?)\s+가격\s*:", content)
                if not match:
                    continue
                name = match.group(1).strip()
                if name not in SANDWICH_MENU_NAMES or name in seen:
                    continue
                seen.add(name)
                menus.append(
                    {
                        "id": _slug(name),
                        "name": name,
                        "aliases": _menu_aliases(name),
                    }
                )
        return sorted(menus, key=lambda item: len(item["name"]), reverse=True)

    # 메뉴 CSV에서 사이즈별 기본 가격표를 만든다.
    @staticmethod
    def _load_prices(path: Path) -> dict[str, dict[str, int]]:
        if not path.exists():
            return {}

        prices: dict[str, dict[str, int]] = {}
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                content = row.get("content", "")
                if "가격" not in content:
                    continue
                name = re.split(r"\s+가격", content, maxsplit=1)[0].strip()
                size_prices: dict[str, int] = {}
                for size, price in re.findall(r"(15cm|30cm)\s*[:：]?\s*([0-9,]+)\s*원", content):
                    size_prices[size] = int(price.replace(",", ""))
                if size_prices:
                    prices[_slug(name)] = size_prices
        return prices

    # 후보 문장에서 등록된 메뉴 별칭과 일치하는 메뉴를 찾는다.
    def _find_menu(self, source_texts: list[str]) -> dict | None:
        for text in source_texts:
            compact = _compact(text)
            for menu in self.menus:
                if any(_compact(alias) in compact for alias in menu["aliases"]):
                    return {"id": menu["id"], "name": menu["name"]}
        return None

    # 후보 문장에서 가장 구체적인 별칭이 일치하는 단일 옵션을 찾는다.
    @staticmethod
    def _find_one(source_texts: list[str], options: list[dict]) -> dict | None:
        for text in source_texts:
            compact = _compact(text)
            for option in sorted(options, key=lambda item: max(len(alias) for alias in item["aliases"]), reverse=True):
                if any(_compact(alias) in compact for alias in option["aliases"]):
                    return option
        return None

    # 후보 문장에서 다중 선택 옵션을 찾고 제외 표현도 함께 처리한다.
    @staticmethod
    def _find_many(source_texts: list[str], options: list[dict]) -> list[dict] | None:
        for text in source_texts:
            compact = _compact(text)
            if not compact:
                continue

            # 먼저 제외된 옵션을 계산해 선택 후보에서 제거한다.
            excluded_ids = {
                option["id"]
                for option in options
                if _option_is_excluded(compact, option)
            }
            values = [
                option
                for option in options
                if option["id"] not in excluded_ids and _option_is_selected(compact, option)
            ]
            if values:
                return values
            # "나머지 전부" 류의 표현은 제외 항목을 뺀 전체 옵션으로 해석한다.
            if excluded_ids and any(word in compact for word in ("나머지", "나머지는", "나머지다", "전부", "다넣", "다넣어")):
                return [option for option in options if option["id"] not in excluded_ids]
            if excluded_ids:
                return None
            
            # 옵션 그룹 자체를 원하지 않는다는 표현은 빈 선택으로 확정한다.
            if _is_empty_option_group(compact):
                return []
        return None

    # 후보 문장에서 토스팅 여부를 True/False로 판정한다.
    @staticmethod
    def _find_toast(source_texts: list[str]) -> bool | None:
        for text in source_texts:
            compact = _compact(text)
            if compact in ("안함", "안", "없음", "미토스팅") or any(word in compact for word in ("안데워", "데우지마", "굽지마", "토스팅안", "그냥", "안구워")):
                return False
            if compact in ("함", "구움") or any(word in compact for word in ("데워", "구워", "토스팅", "따뜻하게")):
                return True
        return None

    # 후보 문장에서 'N개' 형태의 수량을 찾는다.
    @staticmethod
    def _find_quantity(source_texts: list[str]) -> int | None:
        for text in source_texts:
            match = re.search(r"([1-9]\d*)\s*(개|개요|개로|개씩)", text)
            if match:
                return int(match.group(1))
        return None

    # 다중 선택 필드에 선택 또는 제외 결과를 적용한다.
    def _apply_many(self, active_order: dict, field: str, source_texts: list[str], options: list[dict]) -> None:
        # 명시적으로 선택된 값이 있으면 해당 필드를 새 선택 목록으로 교체한다.
        values = self._find_many(source_texts, options)
        if values is not None:
            active_order[field] = [self._ref(value) for value in values]
            return

        # 선택값은 없고 제외 표현만 있으면 현재 선택 목록에서 해당 옵션만 제거한다.
        excluded_ids = self._excluded_option_ids(source_texts, options)
        if not excluded_ids:
            return

        current_values = active_order.get(field, [])
        remaining_values = [
            item for item in current_values
            if item.get("id") and item.get("id") not in excluded_ids
        ]
        active_order[field] = remaining_values or [_empty_ref()]

    # RAG가 반환한 '항목: 값' 형식의 주문 상태를 딕셔너리로 파싱한다.
    @staticmethod
    def _parse_order_state(order_info: str) -> dict[str, str]:
        state: dict[str, str] = {}
        for line in (order_info or "").splitlines():
            match = re.match(r"\s*([^:\n]+)\s*:\s*(.+?)\s*$", line)
            if not match:
                continue
            key = match.group(1).strip()
            value = match.group(2).strip()
            if not value or value == "미정":
                continue
            state[key] = value
        return state

    # 주문 상태값과 현재 사용자 발화를 필드 매칭 후보 문장으로 구성한다.
    @staticmethod
    def _field_sources(
        order_state: dict[str, str],
        state_keys: list[str],
        user_text: str,
        previous_step: str,
        user_text_steps: set[str],
    ) -> list[str]:
        sources = []
        state_value = _state_value(order_state, state_keys)
        if state_value:
            sources.append(state_value)
        if previous_step in user_text_steps:
            sources.append(user_text or "")
        return sources

    # 후보 문장에 포함된 제외 옵션 ID 집합을 찾는다.
    @staticmethod
    def _excluded_option_ids(source_texts: list[str], options: list[dict]) -> set[str]:
        excluded_ids: set[str] = set()
        for text in source_texts:
            compact = _compact(text)
            excluded_ids.update(
                option["id"]
                for option in options
                if _option_is_excluded(compact, option)
            )
        return excluded_ids

    # 메뉴, 사이즈, 추가 항목, 수량을 기준으로 가격 정보를 갱신한다.
    def _apply_price(self, active_order: dict) -> None:
        menu_id = active_order["menu"]["id"]
        size = active_order["size"]
        if not menu_id or not size:
            return

        base_price = self.prices.get(menu_id, {}).get(size)
        if base_price is None:
            return

        # 기본 가격에 유료 사이드/추가 재료 가격을 더한 뒤 수량을 곱한다.
        extra_price = self._selected_item_price(active_order)
        quantity = active_order.get("quantity") or 1
        active_order["price"] = {
            "base_price": base_price,
            "extra_price": extra_price,
            "total_price": (base_price + extra_price) * quantity,
        }

    # 현재 선택된 유료 사이드와 추가 재료의 합산 금액을 계산한다.
    @staticmethod
    def _selected_item_price(active_order: dict) -> int:
        option_prices = {
            option["id"]: option.get("price", 0)
            for option in [*SIDE_OPTIONS, *EXTRA_OPTIONS]
        }
        return sum(
            option_prices.get(item.get("id"), 0)
            for field in ("side_menu_items", "extras")
            for item in active_order.get(field, [])
            if item.get("id")
        )

    # 장바구니의 누락 항목을 기준으로 다음 주문 단계를 반환한다.
    @staticmethod
    def _next_step(active_order: dict) -> str:
        if not active_order["menu"]["id"]:
            return "MENU_SELECT"
        if active_order["size"] is None:
            return "SIZE_SELECT"
        if not active_order["bread"]["id"]:
            return "BREAD_SELECT"
        if not active_order["cheese"]["id"]:
            return "CHEESE_SELECT"
        if active_order["is_toasted"] is None:
            return "TOAST_SELECT"
        if _is_template_list(active_order["vegetables"]):
            return "VEGETABLE_SELECT"
        if _is_template_list(active_order["sauces"]):
            return "SAUCE_SELECT"
        if _is_template_list(active_order["side_menu_items"]):
            return "SIDE_SELECT"
        return "ORDER_CONFIRM"

    # 옵션 객체를 장바구니 저장용 참조 형태로 변환한다.
    @staticmethod
    def _ref(option: dict) -> dict:
        return {"id": option["id"], "name": option["name"]}


SIZE_OPTIONS = [
    {"id": "15cm", "name": "15cm", "aliases": ["15", "15cm", "15센치", "15센티", "작은거"]},
    {"id": "30cm", "name": "30cm", "aliases": ["30", "30cm", "30센치", "30센티", "큰거"]},
]


SANDWICH_MENU_NAMES = {
    "잠봉 플러스",
    "잠봉",
    "머쉬룸",
    "터키",
    "터키 베이컨 아보카도",
    "에그 슬라이스",
    "치킨 슬라이스",
    "치킨 베이컨 아보카도",
    "로스트 치킨",
    "로티세리 바비큐 치킨",
    "베지",
    "에그마요",
    "비엘티",
    "이탈리안 비엠티",
    "참치",
    "스파이시 이탈리안",
    "치킨 데리야끼",
    "쉬림프",
    "스테이크&치즈",
    "스파이시 쉬림프",
    "안창 비프",
    "안창 비프&머쉬룸",
    "써브웨이 클럽",
    "폴드포크",
}

BREAD_OPTIONS = [
    {"id": "white", "name": "화이트", "aliases": ["화이트"]},
    {"id": "wheat", "name": "위트", "aliases": ["위트"]},
    {"id": "parmesan-oregano", "name": "파마산 오레가노", "aliases": ["파마산", "오레가노", "파마산오레가노"]},
    {"id": "honey-oat", "name": "허니오트", "aliases": ["허니오트"]},
    {"id": "flatbread", "name": "플랫브레드", "aliases": ["플랫", "플랫브레드"]},
]

CHEESE_OPTIONS = [
    {"id": "american", "name": "아메리칸 치즈", "aliases": ["아메리칸", "아메리칸치즈"]},
    {"id": "shredded", "name": "슈레드 치즈", "aliases": ["슈레드", "슈레드치즈"]},
    {"id": "mozzarella", "name": "모차렐라 치즈", "aliases": ["모차렐라", "모짜렐라", "모차렐라치즈"]},
]

VEGETABLE_OPTIONS = [
    {"id": "lettuce", "name": "양상추", "aliases": ["양상추"]},
    {"id": "tomato", "name": "토마토", "aliases": ["토마토"]},
    {"id": "cucumber", "name": "오이", "aliases": ["오이"]},
    {"id": "pepper", "name": "피망", "aliases": ["피망", "파프리카"]},
    {"id": "onion", "name": "양파", "aliases": ["양파"]},
    {"id": "pickle", "name": "피클", "aliases": ["피클"]},
    {"id": "olive", "name": "올리브", "aliases": ["올리브"]},
    {"id": "jalapeno", "name": "할라피뇨", "aliases": ["할라피뇨", "할라피노"]},
]

SAUCE_OPTIONS = [
    {"id": "ranch", "name": "랜치", "aliases": ["랜치"]},
    {"id": "sweet-onion", "name": "스위트 어니언", "aliases": ["스위트어니언", "어니언"]},
    {"id": "mayo", "name": "마요네즈", "aliases": ["마요", "마요네즈"]},
    {"id": "sweet-chili", "name": "스위트 칠리", "aliases": ["스위트칠리", "칠리"]},
    {"id": "smoke-bbq", "name": "스모크 바비큐", "aliases": ["스모크바비큐", "바비큐", "바베큐"]},
    {"id": "honey-mustard", "name": "허니 머스타드", "aliases": ["허니머스타드", "머스타드"]},
    {"id": "olive-oil", "name": "올리브 오일", "aliases": ["올리브오일", "오일"]},
    {"id": "salt", "name": "소금", "aliases": ["소금"]},
    {"id": "black-pepper", "name": "후추", "aliases": ["후추"]},
]

SIDE_OPTIONS = [
    {"id": "drink-soda", "name": "탄산음료", "aliases": ["탄산음료", "탄산 음료", "탄산", "음료", "콜라", "코카콜라", "제로콜라", "사이다", "스프라이트"], "price": 2000},
    {"id": "drink-coffee", "name": "커피", "aliases": ["커피", "아메리카노", "아아", "아이스아메리카노"], "price": 2000},
    {"id": "soup-potato-bacon-regular", "name": "포테이토 베이컨 수프 레귤러", "aliases": ["포테이토 베이컨 수프 레귤러", "감자 베이컨 수프 레귤러"], "price": 4100},
    {"id": "soup-potato-bacon-half", "name": "포테이토 베이컨 수프 하프", "aliases": ["포테이토 베이컨 수프", "감자 베이컨 수프", "포테이토 수프", "수프", "스프"], "price": 2500},
    {"id": "soup-corn-regular", "name": "콘 수프 레귤러", "aliases": ["콘 수프 레귤러", "옥수수 수프 레귤러"], "price": 4100},
    {"id": "soup-corn-half", "name": "콘 수프 하프", "aliases": ["콘 수프", "옥수수 수프", "콘 스프"], "price": 2500},
    {"id": "soup-mushroom-regular", "name": "머쉬룸 수프 레귤러", "aliases": ["머쉬룸 수프 레귤러", "버섯 수프 레귤러"], "price": 4100},
    {"id": "soup-mushroom-half", "name": "머쉬룸 수프 하프", "aliases": ["머쉬룸 수프", "버섯 수프", "머쉬룸 스프"], "price": 2500},
    {"id": "cookie-orange-chocolate-chip", "name": "오렌지 초코칩 쿠키", "aliases": ["오렌지 초코칩", "오렌지 초코칩 쿠키"], "price": 1500},
    {"id": "cookie-double-chocolate-chip", "name": "더블 초코칩 쿠키", "aliases": ["더블 초코칩", "더블 초코칩 쿠키"], "price": 1500},
    {"id": "cookie-oatmeal-raisin", "name": "오트밀 레이즌 쿠키", "aliases": ["오트밀 레이즌", "오트밀 쿠키", "레이즌 쿠키"], "price": 1500},
    {"id": "cookie-raspberry-cheesecake", "name": "라즈베리 치즈케익 쿠키", "aliases": ["라즈베리", "라즈베리 쿠키", "라즈베리 치즈케익"], "price": 1500},
    {"id": "cookie-white-choco-macadamia", "name": "화이트 초코 마카다미아 쿠키", "aliases": ["화이트 초코", "마카다미아 쿠키", "화이트 초코 마카다미아"], "price": 1500},
    {"id": "cookie-chocolate-chip", "name": "초코칩 쿠키", "aliases": ["초코칩", "초코칩 쿠키", "쿠키"], "price": 1500},
    {"id": "wedge-potato-bacon-cheesy", "name": "Bacon Cheesy 웨지 포테이토", "aliases": ["베이컨 치지 웨지", "베이컨 치즈 웨지", "bacon cheesy 웨지"], "price": 2900},
    {"id": "wedge-potato-cheesy", "name": "Cheesy 웨지 포테이토", "aliases": ["치지 웨지 포테이토", "치즈 웨지", "cheesy 웨지"], "price": 2500},
    {"id": "wedge-potato", "name": "웨지 포테이토", "aliases": ["웨지 포테이토", "웨지 감자", "감자"], "price": 2000},
]

EXTRA_OPTIONS = [
    {"id": "avocado", "name": "아보카도", "aliases": ["아보카도"], "price": 1500},
    {"id": "bacon", "name": "베이컨", "aliases": ["베이컨"], "price": 1500},
    {"id": "double-meat", "name": "주재료 2배", "aliases": ["주재료2배", "고기추가"], "price": 3000},
]


# 비교를 쉽게 하도록 문자열을 소문자화하고 공백을 제거한다.
def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


# 메뉴명을 장바구니 ID로 사용할 수 있는 슬러그로 변환한다.
def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z가-힣]+", "-", value.lower()).strip("-")
    return slug or f"menu-{uuid4().hex[:8]}"


#  메뉴명별 기본 별칭과 자주 쓰는 대체 표기를 만든다.
def _menu_aliases(name: str) -> list[str]:
    aliases = [name, name.replace(" ", "")]
    if name == "써브웨이 클럽":
        aliases.extend(["서브웨이 클럽", "써브웨이클럽", "서브웨이클럽", "샌드위치 클럽", "클럽"])
    if name == "이탈리안 비엠티":
        aliases.extend(["이탈리안 BMT", "비엠티", "BMT", "bmt"])
    if name == "스테이크&치즈":
        aliases.extend(["스테이크 치즈", "스테이크앤치즈"])
    if name == "안창 비프&머쉬룸":
        aliases.extend(["안창 비프 머쉬룸", "안창비프머쉬룸"])
    return aliases


# 옵션 목록이 아직 템플릿의 미선택 상태인지 확인한다.
def _is_template_list(value: list[dict]) -> bool:
    return len(value) == 1 and value[0].get("id") is None and value[0].get("name") is None


# 미선택 상태로 사용할 빈 옵션 참조를 반환한다.
def _empty_ref() -> dict:
    return {"id": None, "name": None}


# 공백 차이를 무시하고 주문 상태에서 지정한 키의 값을 찾는다.
def _state_value(order_state: dict[str, str], keys: list[str]) -> str | None:
    compact_to_value = {_compact(key): value for key, value in order_state.items()}
    for key in keys:
        value = compact_to_value.get(_compact(key))
        if value:
            return value
    return None


# 정규화된 문장에 옵션 별칭이 포함되어 있는지 확인한다.
def _option_is_selected(compact_text: str, option: dict) -> bool:
    """"""
    return any(_compact(alias) in compact_text for alias in option["aliases"])


# 정규화된 문장에서 옵션 제외 표현이 있는지 확인한다.
def _option_is_excluded(compact_text: str, option: dict) -> bool:
    return any(_alias_is_excluded(compact_text, _compact(alias)) for alias in option["aliases"])


# 옵션 별칭 뒤에 '빼고', '없이' 같은 표현이 붙어서 해당 옵션이 제외되었는지 판정한다.
def _alias_is_excluded(compact_text: str, compact_alias: str) -> bool:
    if not compact_alias:
        return False
    exclusion_suffixes = (
        "빼",
        "빼고",
        "빼줘",
        "빼주세요",
        "제외",
        "제외하고",
        "제외해",
        "제외해주세요",
        "안넣",
        "안넣어",
        "안넣어줘",
        "넣지마",
        "넣지말아",
        "없이",
    )
    if any(f"{compact_alias}{suffix}" in compact_text for suffix in exclusion_suffixes):
        return True
    if any(f"{compact_alias}는{suffix}" in compact_text or f"{compact_alias}은{suffix}" in compact_text for suffix in exclusion_suffixes):
        return True
    return False


# 옵션 그룹 전체를 선택하지 않겠다는 표현인지 확인한다.
def _is_empty_option_group(compact_text: str) -> bool:
    return any(
        word in compact_text
        for word in (
            "없음",
            "없어요",
            "필요없",
            "안넣",
            "안넣어",
            "아무것도안",
            "다빼",
            "전부빼",
            "모두빼",
            "야채없이",
            "채소없이",
            "소스없이",
            "사이드없이",
            "추가없이",
        )
    )


# 로그에 남길 때 줄바꿈을 이스케이프하고 너무 긴 텍스트는 자른다.
def _compact_log_text(value: str | None, limit: int = MAX_CART_LOG_TEXT_CHARS) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
