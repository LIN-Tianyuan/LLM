from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# 定义对话模板 (ChatPromptTemplate)
prompt = ChatPromptTemplate.from_messages(
    [
        # 设定角色
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        # 允许历史消息存储在history变量中
        MessagesPlaceholder(variable_name="history"),
        # 用户输入
        ("human", "{input}"),
    ]
)
# 创建 ChatOpenAI 模型
model = ChatOpenAI(model_name="gpt-4")
runnable = prompt | model

store = {}

REDIS_URL = "redis://localhost:6379/0"

def get_message_history(session_id: str) -> RedisChatMessageHistory:
    # 使用 RedisChatMessageHistory 作为对话历史存储
    # session_id 用于唯一标识一个对话会话
    return RedisChatMessageHistory(session_id, url=REDIS_URL)

# 构建可跟踪历史的 RunnableWithMessageHistory
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_message_history,
    input_messages_key="input", # 用户输入的键是 input
    history_messages_key="history", # 历史消息存储在 history
)
response = with_message_history.invoke(
    {"ability": "math", "input": "余弦是什么意思？"},
    # config 指定 session_id="abc123"，用于存储和检索 Redis 里的对话历史。
    config={"configurable": {"session_id": "abc123"}},
)
print(response)
# content="Cosine is a trigonometric function comparing the ratio of an angle's adjacent side to its hypotenuse in a right triangle." response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 33, 'total_tokens': 60}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-c383660d-7195-4b36-9175-992f05739ece-0' usage_metadata={'input_tokens': 33, 'output_tokens': 27, 'total_tokens': 60}

# 记住
response = with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "abc123"}},
)
print(response)

# 新的 session_id --> 不记得了。
response = with_message_history.invoke(
    {"ability": "math", "input": "什么?"},
    config={"configurable": {"session_id": "def234"}},
)
print(response)
