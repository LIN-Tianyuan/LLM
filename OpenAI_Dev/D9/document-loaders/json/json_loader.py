# pip install jq

# jq 是一个强大的 JSON 处理工具，这里用于提取 JSON 数据。
from langchain_community.document_loaders import JSONLoader

from pprint import pprint

loader = JSONLoader(
    file_path='../../resource/country.json',
    jq_schema='.messages[].content',
    # text_content=False 让 content 仍然是JSON 结构（不转为纯文本）
    text_content=False
)
data = loader.load()
pprint(data)

"""
[Document(metadata={'source': '/Users/citron/Documents/GitHub/LLM/OpenAI_Dev/D9/resource/country.json', 'seq_num': 1}, page_content='{"name": "Item 1", "type": "Type A", "country": "USA", "description": "This is the description of Item 1."}'),
 Document(metadata={'source': '/Users/citron/Documents/GitHub/LLM/OpenAI_Dev/D9/resource/country.json', 'seq_num': 2}, page_content='{"name": "Item 2", "type": "Type B", "country": "Canada", "description": "This is the description of Item 2."}'),
 Document(metadata={'source': '/Users/citron/Documents/GitHub/LLM/OpenAI_Dev/D9/resource/country.json', 'seq_num': 3}, page_content='{"name": "Item 3", "type": "Type C", "country": "UK", "description": "This is the description of Item 3."}'),
 Document(metadata={'source': '/Users/citron/Documents/GitHub/LLM/OpenAI_Dev/D9/resource/country.json', 'seq_num': 4}, page_content='{"name": "Item 4", "type": "Type D", "country": "Germany", "description": "This is the description of Item 4."}')]
"""