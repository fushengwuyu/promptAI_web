# author: sunshine
# datetime:2022/10/19 下午5:55


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

"""
意图分类：
帮我定一个周日上海浦东的房间
选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
答案：
"""


class PromptAI:
    def __init__(self, model_path, gpu_id=-1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device('cpu') if gpu_id == -1 else torch.device(f'cuda:{gpu_id}')

        self.model.to(self.device)

    def answer(self, text, sample=False, top_p=0.6):
        """
        :param text:
        :param sample: 是否抽样。生成任务，可以设置为True;
        :param top_p: top_p：0-1之间，生成的内容越多样、
        :return:
        """
        text = self.preprocess(text)
        encoding = self.tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(
            self.device)
        if not sample:  # 不进行采样
            out = self.model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128,
                                      num_beams=4,
                                      length_penalty=0.6)
        else:  # 采样（生成）
            out = self.model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128,
                                      do_sample=True, top_p=top_p)
        out_text = self.tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
        return self.postprocess(out_text[0])

    def preprocess(self, text):
        return text.replace("\n", "_")

    def postprocess(self, text):
        return text.replace("_", "\n")


if __name__ == '__main__':
    pa = PromptAI(model_path='/media/sunshine/浮生物语的外置空间/pre_models/PromptCLUE-base', gpu_id=-1)
    text = "问答:\n" \
           "问题：中国首都在哪\n" \
           "答案：\n"
    print(pa.answer(text, sample=True))
else:
    promptAI = PromptAI(model_path='/media/sunshine/浮生物语的外置空间/pre_models/PromptCLUE-base', gpu_id=-1)
