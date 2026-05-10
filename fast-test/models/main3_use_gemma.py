import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from enum import Enum, auto
import pandas as pd
import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import re
import warnings
warnings.filterwarnings("ignore")

# from huggingface_hub import login
# load_dotenv()
# login(os.environ.get("HF_TOKEN"), add_to_git_credential=False)
# print("로그인 성공! 이제 모델을 불러옵니다.")

# or huggingface-cli login

# class ProcessStep(Enum):
#     Check_Menu = "메인 메뉴 선택 이전"
#     Subway_Order_Process = "서브웨이 주문 프로세스 진행"
#     PAYMENT = "결제"
#     END = "응대 완료"

class RAGChatbot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        #
        self.model_id = "DimensionSTP/gemma-3-12b-it-Ko-Reasoning"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,       # 중첩 양자화로 메모리 추가 절약
            bnb_4bit_quant_type="nf4",            # 4비트 정밀도 최적화
            bnb_4bit_compute_dtype=torch.bfloat16 # 4080은 bfloat16을 지원하므로 속도 향상
        )
        self.model = None
        self.embed_model = None
        self.processor = None
        self.index_name = "./vectorstore/SUBWAY_MENU.index"
        self.top_k = 3
        #
        self.user_message = None
        self.result_indices = None
        self.result_distances = None
        self.prompts = None
        self.answer = None
        self.docs_search_text = None
        self.retrieved_docs_text = ""
        self.history = None
        self.order_process = None
        self.order_info = ""

    def load_gemma_quant(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def load_embed_model(self):
        self.embed_model = SentenceTransformer("google/embeddinggemma-300m")

    def load_index(self):
        index = faiss.read_index(self.index_name)
        base_name = os.path.basename(self.index_name).split('.')[0]
        df = pd.read_csv(f'./vectorstore/{base_name}.csv', encoding='utf-8-sig')
        self.v_index = index
        self.v_df = df

    def text_to_vector(self, text):
        return self.embed_model.encode_query(text)

    def search(self):
        query_vector = self.text_to_vector(self.docs_search_text)
        query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
        D, I = self.v_index.search(query_vector, self.top_k)
        self.result_distances = D[0]
        self.result_indices = I[0]

    def get_retrieved_docs(self):
        self.search()

        retrieved_docs = []
        for doc_idx in self.result_indices:
            if doc_idx == -1: continue
            doc_text = self.v_df.iloc[doc_idx]['content']
            retrieved_docs.append(f"Document {doc_idx}: {doc_text}")

        separator = "\n"
        self.retrieved_docs_text = separator.join(retrieved_docs)
        # retrieved_docs_text = 합친 내용, retrieved_docs = 내용의 각 원소

    def do_question_process(self):
        inputs = self.processor.apply_chat_template(
            self.prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1] # 질문에 대한 토큰 개수 확인. 답변시 여길 기준으로 자르기
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=200, # 답변 max 토큰 수 => 더 작아야되는데, 이후 테스트 하면서 조정
                do_sample=True,  # Reasoning 모델은 약간의 샘플링이 자연스러울 수 있음
                temperature=0.1 # 답변 창의성: 높을수록 엉뚱한 대답 확률 증가. 이후 테스트 하면서 조정
            )
            generation = generation[0][input_len:] # 방금 계산한 질문 토큰 이후부터 답변 토큰만 출력
        self.answer = self.processor.decode(generation, skip_special_tokens=True)

    def make_prompt_and_answer_gpt(self):
        SYSTEM_PROMPT = "\n".join([
            "[기본 정보]",
            "전체 샌드위치 종류: 잠봉 플러스, 잠봉, 머쉬룸, 터키, 터키 베이컨 아보카도, 에그 슬라이스, 치킨 슬라이스, 치킨 베이컨 아보카도, 로스트 치킨, 로티세리 바비큐 치킨, 베지, 에그마요, 비엘티, 이탈리안 비엠티, 참치, 스파이시 이탈리안, 치킨 데리야끼, 쉬림프, 스테이크&치즈, 스파이시 쉬림프, 안창 비프, 안창 비프&머쉬룸, 써브웨이 클럽, 폴드포크",
            "샌드위치 빵 목록: 화이트, 위트, 파마산 오레가노, 허니오트, 그레인, 플랫브레드",
            "샌드위치 치즈 목록: 아메리칸 치즈, 슈레드 치즈, 모짜렐라 치즈",
            "샌드위치 야채 목록: 양상추, 토마토, 오이, 피망(파프리카), 양파, 피클, 올리브, 할라피뇨, 아보카도",
            "샌드위치 소스 목록: 랜치, 스위트 어니언, 마요네즈, 스위트 칠리, 스모크 바비큐, 핫 칠리, 허니 머스타드, 사우스웨스트 치폴레, 홀스래디쉬, 저당 크리미 어니언, 엑스트라 버진 올리브 오일, 레드 와인 식초, 소금, 후추",
            "샌드위치 추가 주문 가능 음식 재료 목록: 에그마요, 베이컨, 치즈, 에그 슬라이스, 아보카도, 오믈렛, 페퍼로니, 주재료 2배",
            "사이드(스마일썹) 카테고리 메뉴: 포테이토 베이컨 수프, 콘 수프, 머쉬룸 수프, 오렌지 초코칩 쿠키, 초코칩 쿠키, 더블 초코칩 쿠키, 오트밀 레이즌 쿠키, 라즈베리 치즈케익 쿠키, 화이트 초코 마카다미아 쿠키, 웨지 포테이토, Cheesy 웨지 포테이토, Bacon Cheesy 웨지 포테이토",
            "음료 메뉴: 커피, 탄산 음료",
            "──────────────────────────────────────────────────",
            "[참조 정보]",
            self.retrieved_docs_text,
            "──────────────────────────────────────────────────",
            "당신은 서브웨이 키오스크 주문 도우미입니다.",
            "",
            "[목표]: 우수하고 성실한 고객 응대를 통해 고객 주문 완료."
            "",
            "[주문 단계]: 메뉴 선택 전, 샌드위치 선택 완료, 빵 선택 완료, 치즈 선택 완료, 야채 선택 완료, 소스 선택 완료, 추가 재료 선택 완료, 사이드 및 음료 선택 완료",
            "",
            "[대화 이력 분석]: 답변 전, 반드시 대화 이력 전체를 읽고 아래 두 가지를 먼저 파악하세요.",
            "1. 고객 제약 조건 추출:",
            "   - 고객이 언급한 알레르기, 기피 재료, 식이 제한을 모두 수집하세요.",
            "   - 예: '조개 없는 거', '땅콩 알레르기', '채식', '돼지고기 빼줘'",
            "   - 이 제약 조건은 이후 모든 답변에서 절대로 위반되면 안 됩니다.",
            "2. 현재 주문 상태 파악:",
            "   - 고객이 이미 확정한 항목(샌드위치, 빵, 치즈 등)을 파악하세요.",
            "   - 확정된 항목은 다시 묻지 마세요.",
            "",
            "[핵심 규칙]:",
            "1. 제공된 '기본 정보'와 '참조 정보'를 답변에 필요하면 활용하세요.",
            "2. 고객의 제약 조건을 위반하는 메뉴는 절대 추천하거나 언급하지 마세요.",
            "3. 주문 단계를 따르며 대화를 이끌어야 합니다. 기본적으로는 주문 단계를 건너뛰지 않되, 고객이 여러 필요 정보를 한번에 제공하였다면 건너뛰어도 됩니다."
            "4. 만약 대화 이력에서 고객이 메뉴 취소, 혹은 이전 단계 선택을 취소하길 바란다면 주문 단계를 그에 맞게 답변해야 합니다."
            "5. 주문 단계가 완료되지 않았다면, 답변에는 파악한 주문 상태를 목록 형태로 반드시 삽입하세요."
            "6. 주문 상태는 답변의 가장 초기에 입력해야하며, 본 답변과는 '======='로 구분지어야 합니다."
            ""
            "[주문 상태 출력 예시 - 모두 미정인 경우]:",
            "[현재 주문 상태]",
            "메인 메뉴: 미정",
            "빵: 미정",
            "치즈: 미정",
            "야채: 미정",
            "소스: 미정",
            "추가 재료: 미정",
            "사이드 및 음료: 미정",
            "*만약 추가 재료, 사이드 및 음료 항목에서 메뉴를 원하지 않는 경우 '없음'이라고 기입하세요."
            ""
        ])

        user_prompt = "\n".join([
            "[대화 이력]",
            self.history,
            "",
            "[현재 고객 발화]",
            self.user_message,
        ])

        response = self.client.responses.create(
            model="gpt-5.4-mini",
            instructions=SYSTEM_PROMPT,
            input=user_prompt,
        )
        self.answer = response.output_text

    def make_prompt_for_search_decision(self):
        SYSTEM_PROMPT = "\n".join([
            "당신은 서브웨이 키오스크 챗봇의 내부 판단 모듈입니다.",
            "",
            "[목표]: 고객 발화가 벡터 데이터베이스 검색을 필요로 하는지 판단",
            "",
            "[챗봇이 기본적으로 알고 있는 정보]:"
            "전체 샌드위치 종류: 잠봉 플러스, 잠봉, 머쉬룸, 터키, 터키 베이컨 아보카도, 에그 슬라이스, 치킨 슬라이스, 치킨 베이컨 아보카도, 로스트 치킨, 로티세리 바비큐 치킨, 베지, 에그마요, 비엘티, 이탈리안 비엠티, 참치, 스파이시 이탈리안, 치킨 데리야끼, 쉬림프, 스테이크&치즈, 스파이시 쉬림프, 안창 비프, 안창 비프&머쉬룸, 써브웨이 클럽, 폴드포크",
            "샌드위치 빵 목록: 화이트, 위트, 파마산 오레가노, 허니오트, 그레인, 플랫브레드",
            "샌드위치 치즈 목록: 아메리칸 치즈, 슈레드 치즈, 모짜렐라 치즈",
            "샌드위치 야채 목록: 양상추, 토마토, 오이, 피망(파프리카), 양파, 피클, 올리브, 할라피뇨, 아보카도",
            "샌드위치 소스 목록: 랜치, 스위트 어니언, 마요네즈, 스위트 칠리, 스모크 바비큐, 핫 칠리, 허니 머스타드, 사우스웨스트 치폴레, 홀스래디쉬, 저당 크리미 어니언, 엑스트라 버진 올리브 오일, 레드 와인 식초, 소금, 후추",
            "샌드위치 추가 주문 가능 음식 재료 목록: 에그마요, 베이컨, 치즈, 에그 슬라이스, 아보카도, 오믈렛, 페퍼로니, 주재료 2배",
            "사이드(스마일썹) 카테고리 메뉴: 포테이토 베이컨 수프, 콘 수프, 머쉬룸 수프, 오렌지 초코칩 쿠키, 초코칩 쿠키, 더블 초코칩 쿠키, 오트밀 레이즌 쿠키, 라즈베리 치즈케익 쿠키, 화이트 초코 마카다미아 쿠키, 웨지 포테이토, Cheesy 웨지 포테이토, Bacon Cheesy 웨지 포테이토",
            "음료 메뉴: 커피, 탄산 음료",
            "",
            "위 정보 이외의 내용에 대한 질문이 있다면 검색을 수행해야 합니다.",
            "[탐색 가능 정보]: 메뉴별 알레르기, 추천 조합, 가격, 영양정보, 공식 메뉴명",
            "",
            "[출력 규칙]:",
            "1. 반드시 True 또는 False 중 하나만 출력",
            "2. 설명, 이유, 부가 문장 절대 금지",
            "",
            "[예시]:",
            "입력: 참치 칼로리 얼마야 - 출력: True",
            "입력: 단백질 순위 알려줘 - 출력: True",
            "입력: 뭐 추천해? - 출력: Fasle - [이유: 메뉴 추천은 정해진 것이 없기에 검색으로 판단 불가]"
            "입력: 네 그걸로 할게요 - 출력: False",
            "입력: 화이트 빵이랑 오레가노랑 무슨 차이가 있는데? - 출력: False - [이유: 그런 깊은 정보는 실려 있지 않음]"
            "입력: 매운 거 추천해줘 - 출력: True",
            "입력: 아까 알려준 거 중에 뭐가 제일 나아 - 출력: False",
        ])

        user_prompt = "\n".join([
            "[대화 이력]",
            self.history,
            "",
            "[현재 고객 발화]",
            self.user_message,
        ])

        self.prompts = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

    def make_prompt_search(self):
        SYSTEM_PROMPT = "\n".join([
            "당신은 서브웨이 벡터 데이터베이스 검색어 생성 AI입니다.",
            "",
            "목표: 사용자 대화를 분석하여 벡터 데이터베이스에서 관련 문서를 가장 정확히 찾을 수 있는 검색어를 생성하세요.",
            "검색은 벡터 거리 유사도 기반으로 이루어지기에 단어가 완벽히 일치하지 않아도 괜찮습니다."
            "",
            "[벡터 데이터베이스]: 메뉴 종류, 가격, 영양정보, 알레르기, 재료, 추천 조합, 순위 정보 등이 저장",
            "",
            "핵심 규칙:",
            "1. 반드시 '핵심 키워드'만 추출하세요.",
            "2. 불필요한 조사, 문장, 감정 표현, 요청 문장은 제거하세요.",
            "3. 출력은 짧은 '키워드 나열' 형태로 작성하세요.",
            "4. 전달된 주문 단계를 참조하여 사용자의 현재 발화와 주문 단계를 조합해 필요 핵심 검색어를 추출하세요."
            "",
            "검색어 생성 규칙:",
            "1. 메뉴명이 포함된 경우 → '공식 메뉴명 중심'",
            "   예: '터키 샌드위치 가격' → 터키 가격",

            "2. 메뉴명이 불명확한 경우 → '재료 + 특징 조합'",
            "   예: '닭고기 저칼로리' → 치킨 저칼로리",

            "3. 정보 조회 질문 → '메뉴명 + 속성'",
            "   - 가격 → 가격",
            "   - 칼로리 → 열량 또는 칼로리",
            "   - 알레르기 → 알레르기",
            "   예: '참치 칼로리 얼마' → 참치 열량",

            "4. 알레르기 질문 → '알레르기명 + 음식 또는 전체'",
            "   예: '새우 없는 샌드위치' → 새우 알레르기 없는 메뉴",

            "5. 순위/추천 질문 → '조건 + 기준'",
            "   예: '단백질 높은 거' → 단백질 높은 메뉴",

            "6. 복합 질문 → '핵심 키워드 모두 포함'",
            "   예: '터키 가격이랑 칼로리' → 터키 가격 열량",

            "7. 오타/별칭 처리:",
            "- 의미 유지하면서 가장 가까운 공식 키워드로 보정",
            "- 예: 'bmt' → 이탈리안 비엠티",

            "출력 규칙:",
            "1. 키워드만 출력 (문장 금지)",
            "2. 공백으로 구분된 키워드 나열",
            "3. 최대 5개 키워드",
            "4. 중복 단어 금지",
            "",
            "예시:",
            "입력: 치킨 데리야끼 칼로리 알려줘 - 출력: 치킨 데리야끼 열량",
            "입력: 돼지고기 안 들어간 거 뭐 있음 - 출력: 돼지고기 없는 메뉴",
            "입력: 제일 단백질 높은 샌드위치 - 출력: 단백질 높은 메뉴",
            "입력: bmt 얼마야 - 출력: 이탈리안 비엠티 가격",
            "입력: 저칼로리 추천 - 출력: 열량 낮은 메뉴",
        ])

        user_prompt = "\n".join([
            "[대화 이력]",
            self.history,
            "",
            "[주문 정보]",
            self.order_info,
            "",
            "[현재 고객 발화]",
            self.user_message,
        ])

        self.prompts = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

    def make_prompt_condense(self):
        SYSTEM_PROMPT = "\n".join([
            "당신은 서브웨이 키오스크의 음성 안내 텍스트를 생성하는 AI입니다.",
            "",
            "[목표]: AI 접대원의 답변을 받아, 키오스크에서 고객에게 음성으로 읽어줄 짧고 자연스러운 문장으로 변환합니다.",
            "",
            "[참조 정보]:",
            f"{self.retrieved_docs_text}",
            "*이는 벡터 검색을 수행하여 접대원이 정보를 참조했을 경우에만 제공됩니다. 제공되었을 경우 답변에 활용하세요.*"
            "[출력 규칙]:",
            "- 짧고 간결하게 요약하세요.",
            "- 존댓말(~요, ~세요)을 사용하세요.",
            "- 불필요한 설명, 인사말, 부연은 제거하세요.",
            "- 고객이 선택/확인해야 할 정보(빵 종류, 사이즈, 소스 등)는 반드시 포함하세요.",
            "- 질문이 있다면 문장 끝에 한 번만 명확하게 물어보세요.",
            "- 생각 과정 없이 바로 최종 문장만 출력하세요."
        ])
        self.prompts = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": self.answer}]
            }
        ]

    def update_order_process(self):
        steps = ["메인 메뉴", "빵", "치즈", "야채", "소스", "추가 재료", "사이드 및 음료"]
        self.order_process = 0
        for step in steps:
            match = re.search(rf"{step}:\s*(.+)", self.order_info)
            if match:
                value = match.group(1).strip()
                if value and value != "미정":
                    self.order_process += 1
                else:
                    break
            else:
                break

    def message_loop(self):
        while True:
            if self.order_process is None:
                self.history = f"assistant: 안녕하세요, 어떤 메뉴를 주문하시겠어요?"
                print("안녕하세요, 어떤 메뉴를 주문하시겠어요?")

            self.user_message = input("\n질문 입력 (종료: q): ")
            if self.user_message.strip().lower() == "q":
                break
            #
            self.make_prompt_for_search_decision()
            self.do_question_process()
            print(self.answer)
            #
            if "True" in self.answer:
                self.make_prompt_search()
                self.do_question_process()
                self.docs_search_text = self.answer
                self.get_retrieved_docs()
            else:
                self.retrieved_docs_text = ""
            #
            self.make_prompt_and_answer_gpt()
            self.order_info = self.answer.split("=======")[0].strip()
            self.answer = self.answer.split("=======")[1].strip()
            self.update_order_process()
            print(self.order_process)
            print(self.order_info)
            print(self.answer)
            #
            self.make_prompt_condense()
            self.do_question_process()
            print(f"\n요약된 답변: {self.answer}")
            #
            self.history = "\n".join([self.history,
                                      f"user: {self.user_message}",
                                      f"assistant: \n{self.order_info}\n=======\n{self.answer}"])
            #
            if self.order_process == 7:
                print("결제 진행하겠습니다.")
                print(" ... 결제 중 ... ")
                print("결제가 완료되었습니다. 감사합니다.")
                #
                self.order_process = None
                self.user_message = None
                self.result_indices = None
                self.result_distances = None
                self.prompts = None
                self.answer = None
                self.docs_search_text = None
                self.retrieved_docs_text = None
                self.history = None
                self.order_info = ""


if __name__ == "__main__":
    chatbot = RAGChatbot()
    chatbot.load_gemma_quant()
    chatbot.load_embed_model()
    chatbot.load_index()
    chatbot.message_loop()