# author: sunshine
# datetime:2022/10/20 上午11:10
from sanic import Sanic
from sanic.response import json
from sanic_openapi import openapi2_blueprint
from prompt_ai import promptAI

app = Sanic(__name__)
app.blueprint(openapi2_blueprint)

tp_mapping = {
    "新闻分类": ['选项'],
    "意图分类": ['选项'],
    "情感分析": ["选项"],
    "推理关系判断": ["前提", "假设", "选项"],
    "阅读理解": ['段落', "问题"],
    "摘要生成": [],
    "翻译成英文": [],
    "翻译成中文": [],
    "信息抽取": ["问题"],
    "阅读文本抽取关键信息": ["问题"],
    "找出指定的信息": ['问题'],
    "下面句子是否表示了相同的语义": ["文本1", "文本2", "选项"],
    "问题生成": [],
    "代词指向哪个名词短语": ["段落", "问题"],
    "抽取关键词": [],
    "文字中包含了怎样的情感": ['选项'],
    "根据标题生成文章": ["标题"],
    "中心词提取": [],
    "生成与下列文字相同意思的句子": []
}


def make_prompt(tp, **kwargs):
    if "text" in kwargs:
        text = kwargs.pop('text')
    else:
        text = ""
    ks = tp_mapping[tp]
    assert set(kwargs.keys()) == set(ks)
    str1 = f"{tp}：\n{text}\n" if text else f"{tp}:\n"
    for k in tp_mapping[tp]:
        v = kwargs[k]
        if isinstance(v, list):
            v = '，'.join(v)
        str1 += f'{k}：{v}\n'
    if tp == "抽取关键词":
        str1 += "关键词：\n"
    else:
        str1 += "答案：\n"

    return str1


@app.route('/answer', methods=['POST'])
def prompt_answer(request):
    d = request.json
    p_type = d.pop('type')
    prompt_str = make_prompt(p_type, **d)

    nend_sample = True if p_type in ["摘要生成", "问题生成", "根据标题生成文章", "生成与下列文字相同意思的句子"] else False
    result = promptAI.answer(prompt_str, sample=nend_sample)
    return json({"answer": result})


if __name__ == '__main__':
    app.run('0.0.0.0', port=7654)
