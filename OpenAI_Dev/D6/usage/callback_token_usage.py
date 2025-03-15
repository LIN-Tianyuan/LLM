# !pip install -qU langchain-community wikipedia
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI

# 还有⼀些特定于 API 的回调上下⽂管理器，允许跟踪多个调⽤中的令牌使⽤情况。
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
with get_openai_callback() as cb:
    result = llm.invoke("告诉我一个笑话")
    print(cb)

"""
Tokens Used: 61
	Prompt Tokens: 14
		Prompt Tokens Cached: 0
	Completion Tokens: 47
		Reasoning Tokens: 0
Successful Requests: 1
Total Cost (USD): $0.00324

"""