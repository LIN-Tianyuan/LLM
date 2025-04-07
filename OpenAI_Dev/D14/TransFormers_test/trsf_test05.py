from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 设置具体包含 config.json 的目录
model_dir = r"D:\PycharmProjects\day14_demo\my_model_cache\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"  # 替换为实际路径

# 加载模型和分词器
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用加载的模型和分词器创建分类任务的 pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 执行分类任务
output = classifier("你好，我是一款语言模型")
print(output)
