# pip install -qU wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
print(tool.invoke({"query": "langchain"}))

print(f"Name: {tool.name}")
print(f"Description: {tool.description}")
print(f"args schema: {tool.args}")
print(f"returns directly?: {tool.return_direct}")

"""
Page: LangChain
Summary: LangChain is a software framework that helps facilitate the integration of 
Name: wikipedia
Description: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
args schema: {'query': {'description': 'query to look up on wikipedia', 'title': 'Query', 'type': 'string'}}
returns directly?: False
"""