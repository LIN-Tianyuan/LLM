# pip install --quiet langchain_experimental
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()

# 拆分的默认⽅式是基于百分位数。在此⽅法中，计算所有句⼦之间的差异，然后任何⼤于X百分位数的差异都会被拆分。
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=50
)

docs = text_splitter.create_documents([knowledge])
print(docs[0].page_content)
"""
﻿I am honored to be with you today at your commencement from one of the finest universities in the world. I never graduated from college.
"""