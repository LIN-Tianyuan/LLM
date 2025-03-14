from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o")
tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位得力的助手。",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# 构建工具代理
agent = create_tool_calling_agent(llm, tools, prompt)
# 通过传入代理和工具来创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)
response = agent_executor.invoke(
    {"input": "谁执导了2023年的电影《奥本海默》，他多少岁了？"}
)
print(response)
"""
{'input': '谁执导了2023年的电影《奥本海默》，他多少岁了？', 'output': '《奥本海默》（2023年）由克里斯托弗·诺兰执导。克里斯托弗·诺兰出生于1970年7月30日，所以截至2023年，他53岁。'}
"""