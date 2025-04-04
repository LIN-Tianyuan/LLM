# SDK方式调用OpenAI接口
from openai import OpenAI   # OpenAI 是官方的 SDK 客户端（OpenAI Python库，必须先 pip install openai）
# pip install openai
import os   # os 是读取环境变量的库，比如 API 密钥、接口地址等可以放环境变量里
# 从环境变量中读取OPENAI_BASE_URL
# print(os.getenv('OPENAI_BASE_URL'))
# 初始化 OpenAI 服务。
client = OpenAI()   # 这会创建一个 OpenAI 接口客户端：它会默认从环境变量读取你的 OPENAI_API_KEY（也就是你的 OpenAI 密钥）
"""
这一段就是：
  向 ChatGPT 的接口发送一组对话消息
  使用模型是 "gpt-4o"（GPT-4 Omni，2024年新出的多模态模型）
  messages 是一个聊天记录的列表：
    system：系统角色设定（可用于设定助手性格）
    user：用户说的话（"Hello"）
"""
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "assistant"},
        {"role": "user", "content": "Hello"}
    ]
)
# print(completion.choices[0].message)
print(completion.choices[0].message.content)    # 这一步就是从返回结果中提取模型生成的回答并打印出来。