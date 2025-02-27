"""Old version of simple_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# 初始化模型
model = ChatOpenAI(temperature=0.9)

# 第一步：生成孩子的正式名字
first_prompt = PromptTemplate(
    input_variables=['lastname'],
    template="我的邻居姓{lastname}，他生了个儿子，他儿子的名字是（只返回名字，不要加额外解释）"
)
first_chain = LLMChain(llm=model, prompt=first_prompt, output_key="child_name")

# 第二步：基于孩子的名字，生成小名
second_prompt = PromptTemplate(
    input_variables=["child_name"],
    template="邻居的儿子名字叫{child_name}，给他起一个小名（只返回小名）"
)
second_chain = LLMChain(llm=model, prompt=second_prompt, output_key="nickname")

# 组合链，传递数据
overall_chain = SequentialChain(
    chains=[first_chain, second_chain],
    input_variables=["lastname"],
    output_variables=["nickname"],
    verbose=True
)

# 运行链
result = overall_chain({"lastname": "王"})
print("小名:", result["nickname"])
"""