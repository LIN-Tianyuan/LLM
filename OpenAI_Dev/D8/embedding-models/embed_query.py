#pip install langchain-openai

from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

# 将多个文本转换成向量
embeddings = embeddings_model.embed_documents(
    [
        "嗨！",
        "哦，你好！",
        "你叫什么名字？",
        "我的朋友们叫我World",
        "Hello World！"
    ]
)

# 将查询语句转换成向量
embedded_query = embeddings_model.embed_query("对话中提到的名字是什么？")
# 打印部分向量值（前5个数值）
print(embedded_query[:5])
"""
[0.0033300963696092367, -0.009467219933867455, 0.03423014283180237, -0.0012061446905136108, -0.015542615205049515]
"""