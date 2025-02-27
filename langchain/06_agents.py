from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

model = ChatOpenAI(model_name="gpt-4o")

tools = load_tools(['llm-math', 'wikipedia'], llm=model)

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True
)


prompt_template = "中国目前有多少人口"
prompt = PromptTemplate.from_template(prompt_template)
print('prompt-->', prompt)

"""
res = agent.run(prompt)
"""
res = agent.invoke(prompt)
print(res['output'])
"""
截至2024年，中国的人口大约是14.08亿。
"""