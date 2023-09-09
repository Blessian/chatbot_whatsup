from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from pathlib import Path
import torch
import time


"""
1. transformers 설치 필요: https://pypi.org/project/transformers/
2. .pth 파일이 해당 파일과 같은 위치 또는 하위에 위치해야 함
3. 객체 초기화에 시간이 걸리기 때문에 객체 생성을 미리 해놓아야함
4. .answer("질문내용")으로 호출하면 str을 return함
5. 대답을 생성하는 시간을 초과하면(10초) "무슨 뜻인지 모르겠어요"를 return함
"""
class KoGPTChatbot():
    def __init__(self):
        self.set_token()
        self.set_token()
        self.set_tokenizer()
        self.set_model()
        
    def set_token(self):
        self.Q_TKN = "<usr>"
        self.A_TKN = "<sys>"
        self.BOS = "</s>"
        self.EOS = "</s>"
        self.MASK = "<unused0>"
        self.SENT = "<unused1>"
        self.PAD = "<pad>"
    
    def set_tokenizer(self):
        self.koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
            "skt/kogpt2-base-v2",
            bos_token=self.BOS, eos_token=self.EOS, unk_token='<unk>',
            pad_token=self.PAD, mask_token=self.MASK
        )
        
    def set_model(self):
        dir = Path("./")
        model_path = str(next(dir.glob("**/chatbot_gpt2_5.pth")))
        
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        model.load_state_dict(
            torch.load(str(dir.joinpath(str(model_path))), 
                       map_location=torch.device('cpu'))
        )
        self.model = model
        
    def answer(self, question:str, sent="0") -> str:
        with torch.no_grad():
            while True:
                q = question.strip()
                if q == "quit": break
                    
                a = ""
                while True:
                    start_time = time.time()

                    input_ids = torch.LongTensor(self.koGPT2_TOKENIZER.encode(self.Q_TKN + q + self.SENT + sent + self.A_TKN + a)).unsqueeze(dim=0)
                    pred = self.model(input_ids)
                    pred = pred.logits
                    gen = self.koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]

                    if gen == self.EOS: break
                    if time.time() - start_time > 10: print("Time out!"); return "무슨 뜻인지 모르겠어요"

                    a += gen.replace("▁", " ")
                    
                return a.strip()