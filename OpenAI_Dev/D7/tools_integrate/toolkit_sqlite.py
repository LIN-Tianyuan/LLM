from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType

db = SQLDatabase.from_uri("sqlite:///langchain.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
print(toolkit.get_tools())

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    toolkit=toolkit,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS
)
result = agent_executor.invoke("Describe the full_llm_cache table")
print(result)
"""
{'input': 'Describe the full_llm_cache table', 'output': 'The `full_llm_cache` table has the following structure:\n\n- `prompt`: A VARCHAR field that is part of the primary key. It cannot be NULL.\n- `llm`: A VARCHAR field that is also part of the primary key. It cannot be NULL.\n- `idx`: An INTEGER field that is part of the primary key as well. It cannot be NULL.\n- `response`: A VARCHAR field that can contain NULL values.\n\nHere are some sample rows from the `full_llm_cache` table:\n\n| prompt | llm | idx | response |\n|--------|-----|-----|----------|\n| [{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs | {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"], "kwargs": {"max_retries": 2, "model_nam | 0 | {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "output", "ChatGeneration"], "kwargs" |'}
"""
