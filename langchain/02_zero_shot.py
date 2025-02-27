"""
Old version of zero-shot.

from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI


template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate(input_variables=["lastname"], template=template)

prompt_text = prompt.format(lastname='王')

model = ChatOpenAI(model_name="gpt-4o")

result = model.invoke(prompt_text)
print(result.content)
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate(input_variables=["lastname"], template=template)

prompt_text = prompt.format(lastname='王')

model = ChatOpenAI(model_name="gpt-4o")

result = model.invoke(prompt_text)
print(result.content)