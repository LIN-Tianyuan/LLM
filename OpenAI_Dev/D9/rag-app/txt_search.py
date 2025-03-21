import streamlit as st
import tempfile
import os
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

# 设置Streamlit应⽤的⻚⾯标题和布局
st.set_page_config(page_title="Rag Agent", layout="wide")
# 设置应⽤的标题
st.title("Lemon AI Agent")

# 上传txt⽂件，允许上传多个⽂件
uploaded_files = st.sidebar.file_uploader(
    label="Upload txt file", type=["txt"], accept_multiple_files=True
)
# 如果没有上传⽂件，提示⽤户上传⽂件并停⽌运⾏
if not uploaded_files:
    st.info("请先上传按TXT⽂档。")
    st.stop()


# 实现检索器
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # 读取上传的⽂档，并写⼊⼀个临时⽬录
    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir=r"./tmp")
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())
    # 使⽤TextLoader加载⽂本⽂件
    loader = TextLoader(temp_filepath, encoding="utf-8")
    docs.extend(loader.load())
    # 进⾏⽂档分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    # 使⽤OpenAI的向量模型⽣成⽂档的向量表示
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(splits, embeddings)
    # 创建⽂档检索器
    retriever = vectordb.as_retriever()
    return retriever


# 配置检索器
retriever = configure_retriever(uploaded_files)
# 如果session_state中没有消息记录或⽤户点击了清空聊天记录按钮，则初始化消息记录
if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，我是小帽AI助⼿，我可以查询⽂档"}]
    # 加载历史聊天记录
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# 创建检索⼯具
from langchain.tools.retriever import create_retriever_tool

# 创建⽤于⽂档检索的⼯具
tool = create_retriever_tool(
    retriever,
    "⽂档检索",
    "⽤于检索⽤户提出的问题，并基于检索到的⽂档内容进⾏回复.",
)
tools = [tool]
# 创建聊天消息历史记录
msgs = StreamlitChatMessageHistory()
# 创建对话缓冲区内存
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

# 指令模板
instructions = """您是⼀个设计⽤于查询⽂档来回答问题的代理。
您可以使⽤⽂档检索⼯具，并基于检索内容来回答问题。
您可能不查询⽂档就知道答案，但是您仍然应该查询⽂档来获得答案。
如果您从⽂档中找不到任何信息⽤于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
"""

# 基础提示模板
base_prompt_template = """
{instructions}

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to us
e a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
# 创建基础提示模板
base_prompt = PromptTemplate.from_template(base_prompt_template)

# 创建部分填充的提示模板
prompt = base_prompt.partial(instructions=instructions)

# 创建llm
llm = ChatOpenAI()

# 创建react Agent
agent = create_react_agent(llm, tools, prompt)

# 创建Agent执⾏器
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# 创建聊天输⼊框
user_query = st.chat_input(placeholder="请开始提问吧!")

# 如果有⽤户输⼊的查询
if user_query:
    # 添加⽤户消息到session_state
    st.session_state.messages.append({"role": "user", "content": user_query})
    # 显示⽤户消息
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        # 创建Streamlit回调处理器
        st_cb = StreamlitCallbackHandler(st.container())
        # agent执⾏过程⽇志回调显示在Streamlit Container (如思考、选择⼯具、执⾏查询、观察结果等)
        config = {"callbacks": [st_cb]}
        # 执⾏Agent并获取响应
        response = agent_executor.invoke({"input": user_query}, config=config)
        # 添加助⼿消息到session_state
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        # 显示助⼿响应
        st.write(response["output"])
