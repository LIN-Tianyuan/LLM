# 安装所需的库
# pip install --upgrade langchain langchain-community langchainhub langchain-chroma bs4
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入BeautifulSoup库，用于解析HTML内容
import bs4
# 从langchain库中导入创建检索链的方法
from langchain.chains import create_retrieval_chain
# 从langchain库中导入创建文档组合链的方法
from langchain.chains.combine_documents import create_stuff_documents_chain
# 从langchain_chroma库中导入Chroma类，用于向量存储
from langchain_chroma import Chroma
# 从langchain_community库中导入WebBaseLoader类，用于加载网页内容
from langchain_community.document_loaders import WebBaseLoader

"""
爬了一篇博客的内容 → 切成小块 → 用向量数据库存起来 → 用用户提问去检索相关内容 → 再让 GPT 模型结合上下文给出答案
"""

"""
访问博客文章网页
用 BeautifulSoup 挑出主要内容（避免广告、页脚等无关信息）
把网页变成 Document 对象，供后续处理
"""
# 使用WebBaseLoader加载网页内容，指定URL和解析参数
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # 使用 BeautifulSoup 解析 HTML 内容时，用来指定要解析的 HTML 元素的class
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
# 加载文档内容
docs = loader.load()
"""
网页内容太长，不能直接扔给 GPT
所以先用 RecursiveCharacterTextSplitter 按规则切成块
每块 1000 个字符，前后块重叠 200 个字符（保证上下文连贯）
"""
# 创建文本分割器，设定每个块的大小和重叠部分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 将文档分割成多个块
splits = text_splitter.split_documents(docs)
"""
每个文本块通过 OpenAI Embedding 向量化
存入本地的向量数据库 Chroma
然后变成 retriever，提问时会从里面找最相关的文本块
"""
# 创建向量存储，使用分割后的文档和OpenAI的嵌入
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()
"""
定义 GPT 模型使用的提示模板（Prompt）
会自动将用户问题插入 {input}，把检索出的内容插入 {context}
"""
# 定义系统提示，用于问答任务
system_prompt = (
    "您是一个用于问答任务的助手。"
    "使用以下检索到的上下文片段来回答问题。"
    "如果您不知道答案，请说您不知道。"
    "最多使用三句话，保持回答简洁。"
    "\n\n"
    "{context}"
)
# 创建聊天提示模板，包含系统提示和用户输入
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# 创建OpenAI聊天模型
llm = ChatOpenAI()
"""
把：检索器（Retriever）+ 问答链（LLM + Prompt） 组合起来
就成了一个完整的 RAG 模型链
"""
# 创建问答链，使用聊天模型和提示模板
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# 创建检索链，将检索器和问答链结合
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# 调用检索链，输入问题并获取回答
response = rag_chain.invoke({"input": "什么是任务分解？"})
# 打印回答
print(response["answer"])
"""
用户提问：什么是任务分解？
模型从向量库中找相关块 + 把它和 prompt 拼成最终的 Prompt → 发给 OpenAI → 得到简洁回答
"""
response = rag_chain.invoke({"input": "我刚刚问了什么?"})
# 打印出来的结果是错误的，没有上下文记忆
print(response["answer"])
"""
任务分解是将复杂的任务分解成多个较小、简单的步骤的过程。这有助于代理（如自主代理系统）更好地理解如何完成任务，并计划在何时执行这些步骤。任务分解可以通过引导模型逐步思考、使用特定指令或借助人类输入来实现。
您刚刚问了关于一个经典平台游戏中MVC组件如何拆分成单独文件的问题。您还提到了这个游戏有10个关卡，主角是一个名叫马里奥的管道工，可以行走和跳跃，类似于超级马里奥。
"""
