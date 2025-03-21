from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载示例文档
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()

# 从没有词边界的语⾔中分割⽂本
text_splitter = RecursiveCharacterTextSplitter(
    # 设置一个非常小的块的大小，只是为了展示。
    chunk_size=100,
    # 块之间的目标重叠。重叠的块有助于在上下文分割时减少信息丢失。
    chunk_overlap=20,
    # 确定块大小的函数。
    length_function=len,
    # 分隔符列表（默认为["\n\n", "\n", " ", ""]）是否应被解析为正则表达式
    is_separator_regex=False
)


texts = text_splitter.create_documents([knowledge])
print(texts[0])
print(texts[1])