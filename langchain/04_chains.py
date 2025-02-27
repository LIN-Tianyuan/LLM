from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate(input_variables=["lastname"], template=template)
model = ChatOpenAI(model_name="gpt-4o")

chain = prompt | model
print(chain.invoke("王").content)
