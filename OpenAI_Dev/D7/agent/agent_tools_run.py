from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

load_dotenv()

search = TavilySearchResults(max_results=1)

# pip install langchain
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
# pip install faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "wiki_search",
    "搜索维基百科",
)

model = ChatOpenAI(model_name="gpt-4")

tools = [search, retriever_tool]

from langchain import hub
# 获取要使用的提示 - 您可以修改这个！
prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(model, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# print(agent_executor.invoke({"input": "你好"}))
print(agent_executor.invoke({"input": "猫的特征"}))
# print(agent_executor.invoke({"input": "猫的特征?今天上海天气怎么样?"}))
"""
{'input': '猫的特征', 'output': '猫的特征可以从感官、身体结构等方面讨论：\n\n**感官：**\n1. 视觉：猫的夜视能力和追踪视觉上的活动物体相当出色，夜视能力是人类的六倍。强光下，猫会将瞳孔缩得如线般狭小，以减少对视网膜的伤害，但视野会因而缩窄。\n2. 听觉：猫每只耳各有32条独立的肌肉控制耳壳转动，因此双耳可单独朝向不同的音源转动。\n3. 嗅觉：家猫的嗅觉比人类灵敏14倍，鼻腔内有2亿个嗅觉受体。\n4. 味觉：猫不光能感知酸、苦、咸味，选择适合自己口味的食物，还能尝出水的味道。\n5. 触觉：猫的感觉毛主要分布于鼻子两侧、下巴、双眼上方、两颊也有数根。感觉毛可以感受非常微弱的空气波动。\n\n**身体结构：**\n猫的爪子尖锐并有伸缩作用，能够向外露张开或往内缩闭藏起来。猫的趾底有脂肪肉垫，因而行走无声。猫在摩蹭时身上会散发出特别的費洛蒙，留下这些独有的費洛蒙宣誓名其领地，提醒其他猫这是我的地盤。\n\n总体来说，猫是一种灵敏感性和掠食性的动物，视觉、听觉、嗅觉、味觉、触觉等感官都非常敏锐，对环境的感知能力强。而且，猫的爪和鼻子等身体结构也为其生存和运动提供了有力的支持。'}
"""