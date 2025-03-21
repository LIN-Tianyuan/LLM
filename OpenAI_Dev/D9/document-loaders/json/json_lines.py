# pip install jq

# jq 是一个强大的 JSON 处理工具，这里用于提取 JSON 数据。
from langchain_community.document_loaders import JSONLoader

from pprint import pprint

loader = JSONLoader(
    file_path='../../resource/country_lines.json',
    jq_schema='.content',
    # text_content=False 让 content 仍然是JSON 结构（不转为纯文本）
    text_content=False,
    json_lines=True
)
data = loader.load()
pprint(data)

