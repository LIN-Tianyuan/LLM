import requests

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"

# 不使用 Authorization 头以进行匿名访问
response = requests.post(API_URL, json={"inputs": "你好，Hugging Face!"})
print(response.json())
