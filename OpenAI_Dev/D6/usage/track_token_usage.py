# !pip install -qU langchain-openai
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4")
msg = llm.invoke([("human", "最古老的楔形文字的已知例子是什么")])
print(msg.response_metadata)
# {'token_usage': {'completion_tokens': 114, 'prompt_tokens': 25, 'total_tokens': 139}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}

# 许多模型提供程序将令牌使⽤信息作为聊天⽣成响应的⼀部分返回。如果可⽤，这将包含在AIMessage.response_metadata 字段中。