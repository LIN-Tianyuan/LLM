from transformers import pipeline
# 本地加载中文RoBERTa问答模型
qa_pipeline = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")
# 提问
result = qa_pipeline({
    "question": "Hugging Face 是什么？",
    "context": "Hugging Face 是一个自然语言处理平台。"
})
print(result)