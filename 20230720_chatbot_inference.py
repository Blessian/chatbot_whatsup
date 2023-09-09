"""
# 필요사항

- pip install sentencepiece
- all.txt
- chatbot.model
- chatbot.vocab
- chatbot_best.pth
"""
import re
import math
import torch
import sentencepiece as spm
from pathlib import Path
from torch import nn
from torch.nn import Transformer


class TFModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.transformer = Transformer(ninp, nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, num_decoder_layers=nlayers,dropout=dropout)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.pos_encoder_d = PositionalEncoding(ninp, dropout)
        self.encoder_d = nn.Embedding(ntoken, ninp)

        self.ninp = ninp
        self.ntoken = ntoken

        self.linear = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, srcmask, tgtmask, srcpadmask, tgtpadmask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        tgt = self.encoder_d(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder_d(tgt)


        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), srcmask, tgtmask, src_key_padding_mask=srcpadmask, tgt_key_padding_mask=tgtpadmask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Chatbot():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab_size = 20000
        self.set_spm()
        self.set_vocabulary()
        self.set_model(self.vocab_size)
        print("Model Loaded!")

    def set_vocabulary(self):
        path = next(Path("./").glob("**/chatbot.model"))
        vocab = spm.SentencePieceProcessor()
        vocab.load(str(path))
        self.vocab = vocab

    def set_spm(self):
        corpus = str(next(Path("./").glob("**/all.txt")))
        prefix = "chatbot"
        vocab_size = 30000
        spm.SentencePieceTrainer.train(
            f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
            " --model_type=bpe" +
            " --max_sentence_length=999999" + # 문장 최대 길이
            " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
            " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
            " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
            " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
            " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰
        self.spm = spm

    def set_model(self, vocab_size):
        path = next(Path("./").glob("**/chatbot_best.pth"))
        model = TFModel(vocab_size+7, 256, 8, 512, 4, 0.1).to(self.device)
        model.load_state_dict(torch.load(str(path), map_location=torch.device('cpu')))
        self.model = model

    @staticmethod
    def preprocess_sentence(sentence):
        sentence = re.sub(r"([?.!,])", r" \1 ", str(sentence))
        sentence = sentence.strip()
        return sentence

    def evaluate(self, model, sentence):
        MAX_LENGTH = 40
        START_TOKEN = [2]
        END_TOKEN = [3]

        sentence = self.preprocess_sentence(sentence)
        input = torch.tensor([START_TOKEN + self.vocab.encode_as_ids(sentence) + END_TOKEN]).to(self.device)
        output = torch.tensor([START_TOKEN]).to(self.device)
        
        def gen_attention_mask(x):
            mask = torch.eq(x, 0)
            return mask

        # 디코더의 예측 시작
        model.eval()
        for i in range(MAX_LENGTH):
            src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(self.device)
            tgt_mask = model.generate_square_subsequent_mask(output.shape[1]).to(self.device)

            src_padding_mask = gen_attention_mask(input).to(self.device)
            tgt_padding_mask = gen_attention_mask(output).to(self.device)

            predictions = model(input, output, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask).transpose(0,1)
            # 현재(마지막) 시점의 예측 단어를 받아온다.
            predictions = predictions[:, -1:, :]
            predicted_id = torch.LongTensor(torch.argmax(predictions.cpu(), axis=-1))


            # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
            if torch.equal(predicted_id[0][0], torch.tensor(END_TOKEN[0])):
                break

            # 마지막 시점의 예측 단어를 출력에 연결한다.
            # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
            output = torch.cat([output, predicted_id.to(self.device)], axis=1)

        return torch.squeeze(output, axis=0).cpu().numpy()

    def answer(self, sentence):
        prediction = self.evaluate(self.model, sentence)
        predicted_sentence = self.vocab.Decode(list(map(int,[i for i in prediction if i < self.vocab_size+7])))

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence
    

bot = Chatbot()
bot.answer("집에 가고 싶다")
