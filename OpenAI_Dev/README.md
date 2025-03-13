# 一、环境配置
## 1. OpenAI环境变量
Windows 导入环境变量
```bash
setx OPENAI_BASE_URL "https://api.openai.com/v1"
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
Mac 导入环境变量
```bash
export OPENAI_API_KEY='https://api.openai.com/v1'
export OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```
## 2. 相关安装
 - Apifox
 - Postman
 - Python(3.12)
 - Pycharm
# 二、接口
## 1. Model
 - 查询模型列表
## 2. Chat
 - gpt-3.5-turbo
 - gpt-4o
## 3. Audio
 - tts-1（创建语音）
 - whisper-1（创建转录、创建翻译）
## 4. Image
 - dall-e-3
 - 创建图像（文生图）
 - 创建图片编辑（垫图）
 - 创建图像变体（改风格）
## 5. Embeddings
 - text-embedding-ada-002
 - text-embedding-3-large
## 三、接口调用
 - request: req_call.py
 - sdk: embedding/sdk_call.py
 - Embedding: embedding/sdk_text_vector.py
 - 向量数据库文本搜索：embedding/search.py
 - 调用具有视觉的 GPT-4o使用本地图片: vision/online_image_to_text.py
 - JSON 模式: json/json_mode.py
 - seed 可重现输出: seed/seed.py
 - 使用代码统计 token 数量开发控制台循环聊天: tiktoken/count_token.py
 - 实现基于最大 token 数量的消息列表限制带会话长度管理的控制台循环聊天: limit_token.py