from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

# ⾃定义 callback handlers
# 实现了 on_llm_new_token 处理程序，以打印我们刚收到的令牌。
# 然后，我们将⾃定义处理程序作为构造函数回调附加到模型对象上。
class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


prompt = ChatPromptTemplate.from_messages(["给我讲个关于{animal}的笑话，限制20个字"])
# 为启用流式处理，我们在ChatModel构造函数中传入`streaming=True`
# 另外，我们将自定义处理程序作为回调参数的列表传入
model = ChatOpenAI(
    model_name="gpt-4", streaming=True, callbacks=[MyCustomHandler()]
)
chain = prompt | model
response = chain.invoke({"animal": "猫"})
print(response.content)