import requests

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
API_TOKEN = ""  # 替换为你的实际 API Token
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# 发送文本生成请求
response = requests.post(API_URL, headers=headers, json={"inputs": "你好，我是一款语言模型，"})
print(response.json())
