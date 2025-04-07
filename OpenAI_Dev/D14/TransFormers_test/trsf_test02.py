from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载模型和分词器，并保存到指定目录
model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "./my_model_cache/uer/gpt2-chinese-cluecorpussmall"

# 下载模型
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# 下载分词器
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型和分词器已下载到: {cache_dir}")
