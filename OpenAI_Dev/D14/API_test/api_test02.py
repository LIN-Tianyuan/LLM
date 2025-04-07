import requests
#使用API_TOKEN
API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN = ""  # 将 your_api_token_here 替换为你的实际 API Token
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# 发送文本生成请求
response = requests.post(API_URL, headers=headers, json={"inputs": "Hello, I am a language model,"})
print(response.json())

