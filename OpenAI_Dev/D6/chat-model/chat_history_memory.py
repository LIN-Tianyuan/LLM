from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
model = ChatOpenAI(model_name="gpt-4")
# 构建⼀个 Runnable
runnable = prompt | model

store = {}

# 要管理消息历史，我们需要：1. 此Runnable; 2. ⼀个返回 BaseChatMessageHistory 实例的可调⽤对象。
# 构建⼀个名为 get_session_history 的可调⽤对象，引⽤此字典以返回 ChatMessageHistory 实例。
# # 默认情况下，期望配置参数是⼀个字符串 session_id 。可以通过 history_factory_config 关键字参数进⾏调整。
# 使⽤单参数默认值
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 通过在运⾏时向 RunnableWithMessageHistory 传递配置，可以指定可调⽤对象的参数。
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

response = with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    config={"configurable": {"session_id": "abc123"}},
)
print(response)
# content="Cosine is a trigonometric function comparing the ratio of an angle's adjacent side to its hypotenuse in a right triangle." response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 33, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-c383660d-7195-4b36-9175-992f05739ece-0' usage_metadata={'input_tokens': 33, 'output_tokens': 27, 'total_tokens': 60}
# 余弦是一种三角函数，通常表示直角三角形的邻边和斜边的比例。

# 记住
response = with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "abc123"}},
)
print(response)
# "余弦"是数学中的一个概念，表示直角三角形的邻边长度和斜边长度的比例。

# 新的 session_id --> 不记得了。
response = with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "def234"}},
)
print(response)
# 抱歉，我没有理解您的问题。请提供数学相关的问题，我会尽力帮助您。
