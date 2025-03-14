# 流式传输 同步
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4")

chunks = []
for chunk in model.stream("天空是什么颜色？"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)
