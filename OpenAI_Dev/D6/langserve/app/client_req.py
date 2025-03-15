# 跨平台调用 url加invoke
import requests
response = requests.post(
    "http://localhost:8000/openai_ext/invoke",
    json={'input': {'topic': '猫'}}
)
print(response.json())