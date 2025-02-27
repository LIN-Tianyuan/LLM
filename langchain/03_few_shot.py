from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

# 提供示例
examples = [ {"word": "开心", "antonym": "难过"}, {"word": "高", "antonym": "矮"}]
example_template = """ 单词: {word}
反义词: {antonym}\\n
"""

# 定义了每个示例的格式
example_prompt = PromptTemplate(input_variables=["word", "antonym"], template=example_template)

# FewShotPromptTemplate 允许 提供示例数据 让 GPT-4o 学会推测模式
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个单词的反义词",
    suffix="单词: {input}\\n反义词:",
    input_variables=["input"],
    example_separator="\\n")

"""
prefix="给出每个单词的反义词" → 作为开头的说明
examples → 示例部分
suffix="单词: {input}\n反义词:" → 用户输入（要查询的单词）
example_separator="\n" → 示例之间的分隔符
"""

prompt_text = few_shot_prompt.format(input='粗')
# print(prompt_text)

model = ChatOpenAI(model_name="gpt-4o")

result = model.invoke(prompt_text)
print(result.content)