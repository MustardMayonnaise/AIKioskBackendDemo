import importlib.util
import gc
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from types import MethodType
from typing import Any

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
MAX_RAG_LOG_TEXT_CHARS = 1200
ORDER_COMPLETE_PROCESS = 9


# 원본 main3_use_gemma.py는 그대로 두고, 서버용 RAG 대화 흐름을 덧씌우는 adapter이다.
# CLI 중심으로 작성된 RAGChatbot을 HTTP 요청/세션 기반으로 재구성하고 필요한 보정은 런타임에만 적용한다.
class RagChatService:
    # 모델 파일 위치와 공유 봇 인스턴스, 로딩 상태, 세션별 대화 상태 저장소를 준비한다.
    # _lock은 무거운 모델을 한 번만 로드하고 하나의 RAGChatbot 인스턴스를 요청 간 안전하게 재사용하기 위해 둔다.
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.models_dir = base_dir / "models"
        self.source_path = self.models_dir / "main3_use_gemma.py"
        self._bot = None
        self._bot_class = None
        self._bot_module = None
        self._loaded = False
        self._lock = Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    # /chat 요청 한 턴을 처리한다.
    # 원본 message_loop()를 직접 쓰지 않고 같은 내부 함수들을 검색 판단 -> 검색 -> 답변 생성 -> 상태 갱신 순서로 호출한다.
    # 세션 상태를 공유 봇 인스턴스에 잠깐 복원해 실행한 다음 최신 history/order_info/order_process를 다시 세션 저장소에 저장한다.
    # 모델 출력은 '=======' 구분자로 주문 상태와 고객 응답을 나눈다는 전제에 의존하며 주문 완료 시 다음 주문을 위해 내부 상태를 초기화한다.
    def ask(self, session_id: str, user_text: str, request_id: str) -> dict[str, Any]:
        self._ensure_loaded(request_id)
        with self._lock:
            request_start = time.perf_counter()
            bot = self._bot
            session = self._sessions.get(session_id, self._new_session())

            # 저장된 세션 상태를 공유 봇 인스턴스에 복원한다.
            bot.history = session["history"]
            bot.order_process = session["order_process"]
            bot.order_info = session["order_info"]
            bot.user_message = user_text
            logger.info(
                "RAG input received (request_id=%s, session_id=%s, user_chars=%d, history_chars=%d, order_process=%s, order_info_chars=%d, user_text=%s, order_info=%s)",
                request_id,
                session_id,
                len(user_text),
                len(bot.history or ""),
                bot.order_process,
                len(bot.order_info or ""),
                self._compact_text(user_text),
                self._compact_text(bot.order_info),
            )

            timings: dict[str, float] = {}
            retrieved_docs_text = ""
            docs_search_text = ""
            raw_order_info = session["order_info"]
            answer = ""
            search_required = False
            search_decision = ""

            try:
                # 먼저 현재 질문에 문서 검색이 필요한지 모델에게 판단시킨다.
                start = time.perf_counter()
                bot.make_prompt_for_search_decision()
                bot.do_question_process()
                timings["search_decision_ms"] = self._elapsed_ms(start)
                search_decision = bot.answer or ""
                search_required = "True" in search_decision
                logger.info(
                    "RAG search decision completed (request_id=%s, session_id=%s, elapsed_ms=%.2f, search_required=%s, prompt_chars=%d, answer_chars=%d, prompt=%s, answer=%s)",
                    request_id,
                    session_id,
                    timings["search_decision_ms"],
                    search_required,
                    len(bot.prompts or ""),
                    len(search_decision),
                    self._compact_text(bot.prompts),
                    self._compact_text(search_decision),
                )

                # 검색이 필요하면 검색어 생성과 벡터 검색을 차례로 수행한다.
                if search_required:
                    start = time.perf_counter()
                    bot.make_prompt_search()
                    bot.do_question_process()
                    docs_search_text = bot.answer or ""
                    bot.docs_search_text = docs_search_text
                    timings["rag_search_query_ms"] = self._elapsed_ms(start)

                    start = time.perf_counter()
                    bot.get_retrieved_docs()
                    retrieved_docs_text = bot.retrieved_docs_text or ""
                    timings["rag_vector_search_ms"] = self._elapsed_ms(start)
                    timings["rag_search_ms"] = timings["rag_search_query_ms"] + timings["rag_vector_search_ms"]
                    logger.info(
                        "RAG retrieval completed (request_id=%s, session_id=%s, query_ms=%.2f, vector_ms=%.2f, docs_search_chars=%d, retrieved_docs_chars=%d, docs_search_text=%s, retrieved_docs_text=%s)",
                        request_id,
                        session_id,
                        timings["rag_search_query_ms"],
                        timings["rag_vector_search_ms"],
                        len(docs_search_text),
                        len(retrieved_docs_text),
                        self._compact_text(docs_search_text),
                        self._compact_text(retrieved_docs_text),
                    )
                else:
                    # 검색이 불필요한 요청도 동일한 응답 스키마를 유지하도록 타이밍 값을 채운다.
                    bot.retrieved_docs_text = ""
                    timings["rag_search_query_ms"] = 0.0
                    timings["rag_vector_search_ms"] = 0.0
                    timings["rag_search_ms"] = 0.0
                    logger.info(
                        "RAG retrieval skipped (request_id=%s, session_id=%s, search_decision=%s)",
                        request_id,
                        session_id,
                        self._compact_text(search_decision),
                    )

                # 모델 답변에서 주문 상태 영역과 고객에게 보여줄 답변 영역을 분리한다.
                start = time.perf_counter()
                bot.make_prompt_and_answer_gpt()
                combined_answer = bot.answer or ""
                raw_order_info, answer = self._split_order_answer(combined_answer, session["order_info"])
                bot.order_info = raw_order_info
                bot.answer = answer
                timings["rag_answer_ms"] = self._elapsed_ms(start)
                logger.info(
                    "RAG answer generated (request_id=%s, session_id=%s, elapsed_ms=%.2f, prompt_chars=%d, combined_answer_chars=%d, order_info_chars=%d, answer_chars=%d, prompt=%s, combined_answer=%s, parsed_order_info=%s, parsed_answer=%s)",
                    request_id,
                    session_id,
                    timings["rag_answer_ms"],
                    len(bot.prompts or ""),
                    len(combined_answer),
                    len(raw_order_info),
                    len(answer),
                    self._compact_text(bot.prompts),
                    self._compact_text(combined_answer),
                    self._compact_text(raw_order_info),
                    self._compact_text(answer),
                )

                # 주문 진행 단계 갱신 실패는 로그로 남기고 응답 생성 흐름은 유지한다.
                start = time.perf_counter()
                previous_order_process = bot.order_process
                try:
                    bot.update_order_process()
                except Exception:
                    logging.exception(
                        "RAG order process update failed (request_id=%s, session_id=%s)",
                        request_id,
                        session_id,
                    )
                timings["rag_order_update_ms"] = self._elapsed_ms(start)
                logger.info(
                    "RAG order process updated (request_id=%s, session_id=%s, elapsed_ms=%.2f, previous_order_process=%s, next_order_process=%s, order_info_chars=%d, order_info=%s)",
                    request_id,
                    session_id,
                    timings["rag_order_update_ms"],
                    previous_order_process,
                    bot.order_process,
                    len(bot.order_info or ""),
                    self._compact_text(bot.order_info),
                )

                # 고객 응답용 문장을 한 번 더 축약하고 실패하면 원문 답변을 그대로 사용한다.
                start = time.perf_counter()
                raw_answer_before_condense = answer
                try:
                    bot.make_prompt_condense()
                    bot.do_question_process()
                    answer = bot.answer or answer
                except Exception:
                    logging.exception(
                        "RAG answer condensation failed; using raw answer (request_id=%s, session_id=%s)",
                        request_id,
                        session_id,
                    )
                timings["rag_condense_ms"] = self._elapsed_ms(start)
                logger.info(
                    "RAG answer condensed (request_id=%s, session_id=%s, elapsed_ms=%.2f, prompt_chars=%d, raw_answer_chars=%d, condensed_answer_chars=%d, prompt=%s, raw_answer=%s, condensed_answer=%s)",
                    request_id,
                    session_id,
                    timings["rag_condense_ms"],
                    len(bot.prompts or ""),
                    len(raw_answer_before_condense),
                    len(answer),
                    self._compact_text(bot.prompts),
                    self._compact_text(raw_answer_before_condense),
                    self._compact_text(answer),
                )

                # 대화 이력에 이번 턴을 누적한다.
                bot.history = "\n".join(
                    [
                        bot.history or "",
                        f"user: {user_text}",
                        f"assistant: \n{raw_order_info}\n=======\n{answer}",
                    ]
                ).strip()

                # 주문이 완료되면 다음 주문을 받을 수 있도록 주문 진행 상태를 초기화한다.
                if bot.order_process == ORDER_COMPLETE_PROCESS:
                    bot.order_process = None
                    bot.order_info = ""

                # 최신 봇 상태를 세션 저장소에 보존하고 응답 payload를 구성한다.
                self._sessions[session_id] = {
                    "history": bot.history,
                    "order_process": bot.order_process,
                    "order_info": bot.order_info,
                }
                timings["rag_total_ms"] = self._elapsed_ms(request_start)
                logger.info(
                    "RAG output ready (request_id=%s, session_id=%s, total_ms=%.2f, final_order_process=%s, search_required=%s, answer_chars=%d, order_info_chars=%d, history_chars=%d, timings=%s, answer=%s, order_info=%s)",
                    request_id,
                    session_id,
                    timings["rag_total_ms"],
                    bot.order_process,
                    search_required,
                    len(answer),
                    len(raw_order_info),
                    len(bot.history or ""),
                    timings,
                    self._compact_text(answer),
                    self._compact_text(raw_order_info),
                )

                return {
                    "answer": answer,
                    "order_info": raw_order_info,
                    "retrieved_docs_text": retrieved_docs_text,
                    "docs_search_text": docs_search_text,
                    "search_required": search_required,
                    "search_decision": search_decision,
                    "order_process": bot.order_process,
                    "timings": timings,
                }
            except Exception:
                logging.exception(
                    "RAG chat failed (request_id=%s, session_id=%s, user_chars=%d)",
                    request_id,
                    session_id,
                    len(user_text),
                )
                raise

    # 서버 시작 시 RAG 리소스를 미리 올리기 위한 공개 진입점이다.
    # 실제 로딩은 _ensure_loaded()가 담당하므로 이미 로드된 상태에서는 아무 작업도 하지 않는다.
    def preload(self, request_id: str = "startup") -> None:
        self._ensure_loaded(request_id)

    # main3_use_gemma.py의 RAGChatbot과 모델 리소스를 한 번만 로드한다.
    # 로드 직후 원본 객체에 주문 단계/프롬프트 보정을 적용하므로 원본 파일을 수정하지 않아도 서버 요구사항을 반영할 수 있다.
    def _ensure_loaded(self, request_id: str) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            start = time.perf_counter()
            logging.info("Loading RAGChatbot (request_id=%s, source=%s)", request_id, self.source_path)
            load_dotenv(self.base_dir / ".env")
            self._bot_module = self._load_bot_module()
            self._bot_class = self._bot_module.RAGChatbot
            # 원본 모델 코드의 상대 경로 의존성을 맞춘 상태에서 단계별로 무거운 리소스를 로드한다.
            with self._models_cwd():
                load_timings: dict[str, float] = {}
                stage_start = time.perf_counter()
                bot = self._bot_class()
                load_timings["bot_init_ms"] = self._elapsed_ms(stage_start)
                try:
                    # 런타임 주문 흐름을 보정한 뒤 Gemma, 임베딩 모델, FAISS 인덱스를 차례로 준비한다.
                    self._patch_runtime_order_flow(bot)
                    self._configure_quantization_for_offload(bot)
                    stage_start = time.perf_counter()
                    bot.load_gemma_quant()
                    load_timings["gemma_load_ms"] = self._elapsed_ms(stage_start)
                    logger.info(
                        "RAG Gemma model loaded (request_id=%s, elapsed_ms=%.2f, model_id=%s)",
                        request_id,
                        load_timings["gemma_load_ms"],
                        getattr(bot, "model_id", None),
                    )
                    stage_start = time.perf_counter()
                    self._load_embed_model(bot)
                    load_timings["embed_model_load_ms"] = self._elapsed_ms(stage_start)
                    logger.info(
                        "RAG embedding model loaded (request_id=%s, elapsed_ms=%.2f)",
                        request_id,
                        load_timings["embed_model_load_ms"],
                    )
                    stage_start = time.perf_counter()
                    bot.load_index()
                    load_timings["index_load_ms"] = self._elapsed_ms(stage_start)
                    logger.info(
                        "RAG FAISS index loaded (request_id=%s, elapsed_ms=%.2f, index_name=%s)",
                        request_id,
                        load_timings["index_load_ms"],
                        getattr(bot, "index_name", None),
                    )
                except Exception:
                    # 일부 리소스만 로드된 상태에서 실패하면 메모리와 CUDA 캐시를 정리한다.
                    self._release_partial_bot(bot)
                    raise
            self._bot = bot
            self._loaded = True
            load_timings["total_load_ms"] = self._elapsed_ms(start)
            logging.info(
                "RAGChatbot loaded (request_id=%s, total_ms=%.2f, timings=%s)",
                request_id,
                load_timings["total_load_ms"],
                load_timings,
            )

    # models/main3_use_gemma.py를 파일 경로로 동적 import한다.
    # spec_from_file_location을 쓰기에 원본 스크립트 위치를 유지한 채 adapter에서 필요한 클래스만 가져온다.
    def _load_bot_module(self):
        spec = importlib.util.spec_from_file_location("set_rag_chatbot", self.source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import RAGChatbot from {self.source_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # 원본 봇의 bitsandbytes 설정 객체에 CPU offload 옵션을 런타임으로 보강한다.
    # VRAM이 부족한 환경을 고려한 설정이다.
    @staticmethod
    def _configure_quantization_for_offload(bot: Any) -> None:
        bnb_config = getattr(bot, "bnb_config", None)
        if bnb_config is None or not hasattr(bnb_config, "llm_int8_enable_fp32_cpu_offload"):
            return
        bnb_config.llm_int8_enable_fp32_cpu_offload = True
        logging.info("Enabled RAG quantization CPU offload for auto device mapping.")

    # 원본 load_embed_model() 대신 Hugging Face 토큰을 명시적으로 넘겨 임베딩 모델을 로드한다.
    # gated model 접근 권한 문제는 원본 예외를 그대로 노출하지 않고 .env에 필요한 토큰 안내가 보이도록 다시 감싼다.
    def _load_embed_model(self, bot: Any) -> None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        sentence_transformer = self._bot_module.SentenceTransformer
        try:
            bot.embed_model = sentence_transformer("google/embeddinggemma-300m", token=token)
        except Exception as exc:
            message = str(exc)
            if "gated repo" in message.lower() or "403" in message:
                raise RuntimeError(
                    "Cannot load google/embeddinggemma-300m because Hugging Face access is restricted. "
                    "Request access to the model and add HF_TOKEN or HUGGINGFACE_HUB_TOKEN to fast-test/.env."
                ) from exc
            raise

    # 원본 RAGChatbot 인스턴스의 주문 흐름만 런타임에서 패치한다.
    # update_order_process는 사이즈/토스팅을 포함한 계산 함수로 교체하고 GPT 주문 프롬프트는 OpenAI 호출 직전에만 보정한다.
    def _patch_runtime_order_flow(self, bot: Any) -> None:
        bot.update_order_process = MethodType(self._update_order_process_with_toasting, bot)
        self._patch_gpt_order_prompt(bot)
        logger.info("Patched RAG order flow with toasting step.")

    # 주문 정보 텍스트에서 확정된 항목 수를 순서대로 계산한다.
    # 기존 7단계 흐름에 사이즈와 토스팅을 추가해 chat_cart_service가 기대하는 단계와 RAG 상태가 어긋나지 않게 맞춘다.
    @staticmethod
    def _update_order_process_with_toasting(bot: Any) -> None:
        steps = ["메인 메뉴", "사이즈", "빵", "치즈", "토스팅", "야채", "소스", "추가 재료", "사이드 및 음료"]
        bot.order_process = 0
        for step in steps:
            match = re.search(rf"{step}:\s*(.+)", bot.order_info)
            if match:
                # 미정 항목을 만나면 이후 단계는 아직 진행하지 않은 것으로 본다.
                value = match.group(1).strip()
                if value and value != "미정":
                    bot.order_process += 1
                else:
                    break
            else:
                break

    # 원본 OpenAI responses.create 호출을 감싸서 주문 프롬프트만 보정한다.
    # 검색 판단/검색어 생성/축약 프롬프트에는 영향을 주지 않도록 instructions에 [주문 단계]가 있을 때만 치환한다.
    def _patch_gpt_order_prompt(self, bot: Any) -> None:
        responses = getattr(getattr(bot, "client", None), "responses", None)
        if responses is None or not hasattr(responses, "create"):
            return

        original_create = responses.create

        # 주문 단계 프롬프트에만 보정 규칙을 적용하고 원래 호출로 위임한다.
        def create_with_toasting_prompt(*args, **kwargs):
            instructions = kwargs.get("instructions")
            if isinstance(instructions, str) and "[주문 단계]" in instructions:
                kwargs["instructions"] = self._add_toasting_prompt(instructions)
            return original_create(*args, **kwargs)

        responses.create = create_with_toasting_prompt

    # 원본 한국어 주문 프롬프트 문자열에 사이즈/토스팅 단계와 기록 규칙을 삽입한다.
    # 단순 문자열 치환 방식이므로 원본 프롬프트 문구가 바뀌면 이 함수의 replacement도 함께 점검해야 한다.
    @staticmethod
    def _add_toasting_prompt(instructions: str) -> str:
        replacements = {
            "[주문 단계]: 메뉴 선택 전, 샌드위치 선택 완료, 빵 선택 완료, 치즈 선택 완료, 야채 선택 완료, 소스 선택 완료, 추가 재료 선택 완료, 사이드 및 음료 선택 완료":
                "[주문 단계]: 메뉴 선택 전, 샌드위치 선택 완료, 사이즈 선택 완료, 빵 선택 완료, 치즈 선택 완료, 토스팅 여부 선택 완료, 야채 선택 완료, 소스 선택 완료, 추가 재료 선택 완료, 사이드 및 음료 선택 완료",
            "   - 고객이 이미 확정한 항목(샌드위치, 빵, 치즈 등)을 파악하세요.":
                "   - 고객이 이미 확정한 항목(샌드위치, 사이즈, 빵, 치즈, 토스팅 여부 등)을 파악하세요.",
            "메인 메뉴: 미정\n빵: 미정":
                "메인 메뉴: 미정\n사이즈: 미정\n빵: 미정",
            "치즈: 미정\n야채: 미정":
                "치즈: 미정\n토스팅: 미정\n야채: 미정",
        }
        for old, new in replacements.items():
            instructions = instructions.replace(old, new)

        # 추가 규칙은 중복 삽입을 피하면서 기존 marker 바로 앞에 배치한다.
        marker = "*만약 추가 재료, 사이드 및 음료 항목에서 메뉴를 원하지 않는 경우 '없음'이라고 기입하세요."
        extra_rules = [
            "사이즈는 메인 메뉴 선택 직후 반드시 15cm 또는 30cm 중 하나로 확인하고, 그 다음 빵 종류를 확인하세요.",
            "토스팅 여부는 치즈 선택 후 반드시 확인하고, 고객이 빵을 구워 달라면 '토스팅: 함', 굽지 않겠다고 하면 '토스팅: 안 함'으로 기입하세요.",
        ]
        for rule in extra_rules:
            if rule not in instructions:
                instructions = instructions.replace(marker, f"{rule}\n{marker}")
        return instructions

    # 로딩 중간 실패 시 이미 잡힌 모델 참조와 CUDA 캐시를 정리한다.
    # Gemma/processor/embed_model 중 일부만 올라간 상태로 남지 않게 해 다음 로딩 시도를 방해하지 않도록 한다.
    def _release_partial_bot(self, bot: Any) -> None:
        if bot is None:
            return
        bot.model = None
        bot.processor = None
        bot.embed_model = None
        gc.collect()
        torch_module = getattr(self._bot_module, "torch", None)
        cuda_module = getattr(torch_module, "cuda", None)
        if cuda_module is not None and cuda_module.is_available():
            cuda_module.empty_cache()

    # 원본 모델 코드 실행 중에만 작업 디렉터리를 models 폴더로 바꾼다.
    # main3_use_gemma.py가 ./vectorstore 같은 상대 경로를 사용하므로, 로딩 구간에서만 cwd를 맞추고 즉시 복구한다.
    @contextmanager
    def _models_cwd(self):
        previous = Path.cwd()
        os.chdir(self.models_dir)
        try:
            yield
        finally:
            os.chdir(previous)

    # 새 session_id가 들어왔을 때 사용할 초기 대화 상태를 만든다.
    # 이후 요청부터는 _sessions에 저장된 history/order_process/order_info를 이어서 사용한다.
    @staticmethod
    def _new_session() -> dict[str, Any]:
        return {
            "history": "assistant: 안녕하세요. 어떤 메뉴를 주문하시겠어요?",
            "order_process": None,
            "order_info": "",
        }

    # RAG가 만든 combined answer를 주문 상태와 고객 응답 문장으로 분리한다.
    # 구분자 누락 시에는 주문 상태를 덮어쓰지 않아 잘못된 모델 출력이 세션 상태를 망가뜨리는 범위를 줄인다.
    @staticmethod
    def _split_order_answer(combined_answer: str, fallback_order_info: str) -> tuple[str, str]:
        if "=======" not in combined_answer:
            return fallback_order_info, combined_answer.strip()
        order_info, answer = combined_answer.split("=======", 1)
        return order_info.strip(), answer.strip()

    # 단계별 성능 로그에 사용할 경과 시간을 밀리초 단위로 계산한다.
    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000

    # 긴 프롬프트/답변을 로그에 남길 때 줄바꿈을 이스케이프하고 최대 길이를 제한한다.
    # 실제 길이는 *_chars 필드로 별도 기록하므로 로그 본문은 추적 가능한 수준까지만 보존한다.
    @staticmethod
    def _compact_text(value: Any, limit: int = MAX_RAG_LOG_TEXT_CHARS) -> str:
        if value is None:
            return ""
        text = str(value).replace("\r", "\\r").replace("\n", "\\n")
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
