import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

class LLMService:
    def __init__(self):
        self.model_id = "DimensionSTP/gemma-3-12b-it-Ko-Reasoning"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,       # 중첩 양자화로 메모리 추가 절약
            bnb_4bit_quant_type="nf4",            # 4비트 정밀도 최적화
            bnb_4bit_compute_dtype=torch.bfloat16 # 4080은 bfloat16을 지원하므로 속도 향상
        )
        self.model = None
        self.processor = None

    def load_gemma_quant(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def question(self, prompts):
        inputs = self.processor.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1] # 질문에 대한 토큰 개수 확인. 답변시 여길 기준으로 자르기

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=256, # 답변 max 토큰 수 => 더 작아야되는데, 이후 테스트 하면서 조정
                do_sample=True,  # Reasoning 모델은 약간의 샘플링이 자연스러울 수 있음
                temperature=0.5 # 답변 창의성: 높을수록 엉뚱한 대답 확률 증가. 이후 테스트 하면서 조정
            )
            generation = generation[0][input_len:] # 방금 계산한 질문 토큰 이후부터 답변 토큰만 출력

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded # answer