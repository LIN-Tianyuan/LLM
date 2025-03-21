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
## 四、LangChain
LangChain 是⼀个⽤于开发由⼤型语⾔模型（LLMs）驱动的应⽤程序的框架。

LangChain 简化了LLM应⽤程序⽣命周期的每个阶段：
 - 开发：使⽤LangChain的开源构建模块和组件构建您的应⽤程序。利⽤第三⽅集成和模板快速
启动。
 - ⽣产部署：使⽤LangSmith检查、监控和评估您的链，以便您可以持续优化并⾃信地部署。
 - 部署：使⽤LangServe将任何链转换为API。
### 1. LCEL(D5/LCEL)
 - Runnable interface
 - Stream
### 2. LLM apps debug: LangSmith Tracing & Verbose, Debug Mode(D5/debug)
 - LangSmith Tracing:debug/lang_smith.py
 - https://smith.langchain.com/public/a89ff88f-9ddc-4757-a395-3a1b365655bf/r
 - 导入环境变量
```bash
#windows导⼊环境变量
setx LANGCHAIN_TRACING_V2 "true"
setx LANGCHAIN_API_KEY "..."
setx TAVILY_API_KEY "..."

#mac 导⼊环境变量
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
export TAVILY_API_KEY="..."
```
### 3. Start LangSmith(D5/debug/lang_smith.py)
 - Create new langsmith api key
 - Install dependencies
```bash
pip install -U langchain langchain-openai
```
 - Configure environment to connect to LangSmith.
```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<>"
LANGSMITH_PROJECT="pr-standard-loincloth-39"
OPENAI_API_KEY="<your-openai-api-key>"
```
 - Run any LLM, Chat model, or Chain. Its trace will be sent to this project.
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")
```
### 4. LangServe(D6/langserve)
LangServe 🦜 🏓 帮助开发者将 LangChain 可运⾏和链部署为 REST API。

该库集成了 FastAPI 并使⽤ pydantic 进⾏数据验证。

从 LangChain 对象**⾃动推断输⼊和输出模式**，并在每次 API 调⽤中执⾏，提供丰富的错误信息

#### 4.1 安装
```bash
pip install "langserve[all]"
```
或者对于客户端代码， pip install "langserve[client]" ，对于服务器代码， pip in
stall "langserve[server]" 。

#### 4.2 LangChain CLI
使⽤ LangChain CLI 快速启动 LangServe 项⽬。

要使⽤ langchain CLI，请确保已安装最新版本的 langchain-cli 。您可以使⽤ pip inst
all -U langchain-cli 进⾏安装。

```bash
pip install -U langchain-cli

langchain -v # 查看版本号
```
#### 4.3 设置
使⽤ poetry 进⾏依赖管理。
 - 使⽤ langchain cli 命令创建新应⽤
```bash
langchain app new my-app
```
 - 在 add_routes 中定义可运⾏对象。转到 server.py 并编辑
```bash
add_routes(app. NotImplemented)
```
 - 使⽤ poetry 添加第三⽅包（例如 langchain-openai、langchain-anthropic、
   langchain-mistral 等）
```bash
# 安装pipx，参考：https://pipx.pypa.io/stable/installation/
pip install pipx
# 加⼊到环境变量，需要重启PyCharm 
pipx ensurepath
# 安装poetry，参考：https://python-poetry.org/docs/
pipx install poetry
# 安装 langchain-openai 库，例如：poetry add [package-name]
poetry add langchain-openai
```
 - 设置相关环境变量。例如:
```bash
export OPENAI_API_KEY="sk-..."
```
 - 启动您的应⽤
```bash
poetry run langchain serve --port=8000
```
#### 4.4 应用
 - 服务器

部署 OpenAI 聊天模型，讲述有关特定主题笑话的链的服务器。
```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
app = FastAPI(
   title="LangChain 服务器",
   version="1.0",
   description="使⽤ Langchain 的 Runnable 接⼝的简单 API 服务器",
)

add_routes(
   app,
   ChatOpenAI(model_name="gpt-4"),
   path="/openai",
)

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="localhost", port=8000)
```
从浏览器调⽤您的端点，还需要设置 CORS 头。(使⽤ FastAPI 的内置中间件来实现)
```python
from fastapi.middleware.cors import CORSMiddleware
# 设置所有启⽤ CORS 的来源
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
   expose_headers=["*"],
)
```
 - 文档

⽂档地址：http://localhost:8000/docs
 - 客户端

Python SDK
```python
from langchain.schema.runnable import RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

openai = RemoteRunnable("http://localhost:8000/openai/")
prompt = ChatPromptTemplate.from_messages([("system", "你是⼀个喜欢写故事的助⼿"), ("system", "写⼀个故事，主题是： {topic}")])
# 可以定义⾃定义链
chain = prompt | RunnableMap({
 "openai": openai
})
response = chain.batch([{"topic": "猫"}])
print(response)
```
在 TypeScript 中（需要 LangChain.js 版本 0.0.166 或更⾼）：
```typescript
import { RemoteRunnable } from "@langchain/core/runnables/remote";
const chain = new RemoteRunnable({
 url: `http://localhost:8000/openai/`,
});
const result = await chain.invoke({
 topic: "cats",
});
```
使⽤ requests 的 Python 代码：
```python
import requests
response = requests.post(
   "http://localhost:8000/openai",
   json={'input': {'topic': 'cats'}}
)
response.json()
```
curl
```bash
curl --location --request POST 'http://localhost:8000/openai/stream' \
   --header 'Content-Type: application/json' \
   --data-raw '{
      "input": {
      "topic": "狗"
      }
   }'
```
 - 端点
```python
...
add_routes(
   app,
   runnable,
   path="/my_runnable",
)
```
将以下端点添加到服务器：
 - POST /my_runnable/invoke - 对单个输⼊调⽤可运⾏项
 - POST /my_runnable/batch - 对⼀批输⼊调⽤可运⾏项
 - POST /my_runnable/stream - 对单个输⼊调⽤并流式传输输出
 - POST /my_runnable/stream_log - 对单个输⼊调⽤并流式传输输出，包括⽣成的中间步骤的输出
 - POST /my_runnable/astream_events - 对单个输⼊调⽤并在⽣成时流式传输事件，包括来⾃中间步骤的事件。
 - GET /my_runnable/input_schema - 可运⾏项的输⼊的 JSON 模式
 - GET /my_runnable/output_schema - 可运⾏项的输出的 JSON 模式
 - GET /my_runnable/config_schema - 可运⾏项的配置的 JSON 模式

这些端点与LangChain 表达式语⾔接⼝相匹配

 - Playground

访问：http://localhost:8000/openai/playground

可以在 /my_runnable/playground/ 找到⼀个可运⾏项的游乐场⻚⾯。这提供了⼀个简单的 UI来配置并调⽤可运⾏项，具有流式输出和中间步骤。

### 5. MessageHistory
#### 5.1 为 Chain 添加 Message history (Memory)单⾏初始化 chat model
(D6/chat-model/chat_history_memory.py)

对话状态Chain传递
 - 在构建聊天机器⼈时，将对话状态传递到链中以及从链中传出对话状态⾄关重要。 
 - RunnableWithMessageHistory 类让我们能够向某些类型的链中添加消息历史。它包装另⼀个 Runnable 并管理其聊天消息历史。
 - 具体来说，它可⽤于任何接受以下之⼀作为输⼊的 Runnable：
   - ⼀系列 BaseMessages
   - 具有以序列 BaseMessages 作为值的键的字典
   - 具有以字符串或序列 BaseMessages 作为最新消息的值的键和⼀个接受历史消息的单独键
     的字典
 - 并将以下之⼀作为输出返回：
   - 可视为 AIMessage 内容的字符串
   - ⼀系列 BaseMessage
   - 具有包含⼀系列 BaseMessage 的键的字典

聊天历史存储在内存
 - 我们构建⼀个名为 get_session_history 的可调⽤对象，引⽤此字典以返回 ChatMessageHistory 实例。
 - 通过在运⾏时向 RunnableWithMessageHistory 传递配置，可以指定可调⽤对象的参数。
 - 默认情况下，期望配置参数是⼀个字符串 session_id 。可以通过 history_factory_config 关键字参数进⾏调整。

#### 5.2 基于 LangChain 的 Chatbot: Chat History
(D6/chat-model/chat_history_config.py)

配置会话唯⼀键
 - 我们可以通过向 history_factory_config 参数传递⼀个 ConfigurableFieldSpec 对象列表
 - 来⾃定义跟踪消息历史的配置参数。
 - 下⾯我们使⽤了两个参数： user_id 和 conversation_id 。
 - 配置user_id和conversation_id作为会话唯⼀键

#### 5.3 消息持久化
(D6/chat-model/chat_history_redis.py)

配置redis环境
```bash
pip install --upgrade --quiet redis
```
如果我们没有现有的 Redis 部署可以连接，可以启动本地 Redis Stack 服务器：
```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```
```python
REDIS_URL = "redis://localhost:6379/0"
```
### 6. 使用LangSmith记录追踪
 - 设置环境变量即可
```
# windows导⼊环境变量
setx LANGCHAIN_TRACING_V2 "true"
setx LANGCHAIN_API_KEY "..."

# mac 导⼊环境变量
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```
### 7. 修剪聊天历史
裁剪消息(chatbot/chatbot_clear_history.py)
 - LLM 和聊天模型有限的上下⽂窗⼝，即使您没有直接达到限制，您可能也希望限制模型处理的⼲扰量。⼀种解决⽅案是只加载和存储最近的 n 条消息。让我们使⽤⼀个带有⼀些预加载消息的示例历史记录。
总结记忆(chatbot/chatbot_summarize_history.py)
 - 使⽤额外的LLM调⽤来在调⽤链之前⽣成对话摘要。
### 8. Track token usage Cache model responses
Track token usage(跟踪token使⽤情况)
 - 使⽤ AIMessage.response_metadata
   - 许多模型提供程序将令牌使⽤信息作为聊天⽣成响应的⼀部分返回。如果可⽤，这将包含在
AIMessage.response_metadata 字段中。
 - 使⽤回调
   - 还有⼀些特定于 API 的回调上下⽂管理器，允许跟踪多个调⽤中的令牌使⽤情况。⽬前仅为
     OpenAI API 和 Bedrock Anthropic API 实现了此功能。
## 五、Multimode 集成
### 1. 多模态数据数据传输
将要求模型描述⼀幅图像。(multimode)

最常⽀持的传⼊图像的⽅式是将其作为字节字符串传⼊。(multimode/multimode_image_base64.py)

传⼊多幅图像(multimode/multimode_image_list.py)

### 2. ⼯具调⽤(multimode/tools_call.py)
## 六、Output parsers: JSON, XML, YAML
output-parser
## 七、自定义Tools，调⽤ Tools集成内建 Tools
### 1. 自定义Tools(tools)
LangChain 提供了三种创建⼯具的⽅式：
1. 使⽤@tool装饰器 -- 定义⾃定义⼯具的最简单⽅式。
2. 使⽤StructuredTool.from_function 类⽅法 -- 这类似于 @tool 装饰器，但允许更多配置和同步和异步实现的规范。
3. 通过⼦类化BaseTool -- 这是最灵活的⽅法，它提供了最⼤程度的控制，但需要更多的⼯作量和代码。 @tool 或 StructuredTool.from_function 类⽅法对于⼤多数⽤例应该⾜够了。 提示 如果⼯具具有精⼼选择的名称、描述和 JSON 模式，模型的性能会更好。
### 2. 调⽤ Tools集成内建 Tools(tools_integrate)
https://python.langchain.com/v0.2/docs/integrations/tools/
## 八、创建和运行 Agent
单独来说，语⾔模型⽆法采取⾏动 - 它们只能输出⽂本。

LangChain 的⼀个重要⽤例是创建代理。

代理是使⽤ LLM 作为推理引擎的系统，⽤于确定应采取哪些⾏动以及这些⾏动的输⼊应该是什
么。

然后可以将这些⾏动的结果反馈给代理，并确定是否需要更多⾏动，或者是否可以结束。

构建⼀个可以与多种不同⼯具进⾏交互的代理：⼀个是本地数据库，另⼀个是搜索引擎。您将能够向该代理提问，观察它调⽤⼯具，并与它进⾏对话。

 - 使⽤语⾔模型，特别是它们的⼯具调⽤能⼒
 - 创建检索器以向我们的代理公开特定信息
 - 使⽤搜索⼯具在线查找信息
 - 聊天历史，允许聊天机器⼈“记住”过去的交互，并在回答后续问题时考虑它们。
 - 使⽤LangSmith调试和跟踪应⽤程序
### 1. 安装LangChain
```bash
pip install langchain
```
### 2. LangSmith
```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```
### 3. 定义工具
Tavily（⽤于在线搜索），然后是我们将创建的本地索引上的检索器。
#### 3.1 Tavily(⽤于在线搜索)
LangChain 中有⼀个内置⼯具，可以轻松使⽤ Tavily 搜索引擎作为⼯具。
```bash
export TAVILY_API_KEY="..."
```
```python
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=2)
print(search.invoke("今天上海天⽓怎么样"))

"""
[{'url': 'http://sh.cma.gov.cn/sh/tqyb/jrtq/', 'content': '上海今天⽓温度30℃
～38℃，偏南⻛⻛⼒4-5级，有多云和雷阵⾬的可能。⽣活⽓象指数显示，⽓温⾼，⼈体感觉不舒适，
不适宜户外活动。'}]
"""
```
#### 3.2 Retriever
Retriever 是 langchain 库中的⼀个模块，⽤于检索⼯具。检索⼯具的主要⽤途是从⼤型⽂本集合或知识库中找到相关信息。它们通常⽤于问答系统、对话代理和其他需要从⼤量⽂本数据中提取信息的应⽤程序。
```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
 # chunk_size 参数在 RecursiveCharacterTextSplitter 中⽤于指定每个⽂档块的最⼤字符数。它的作⽤主要有以下⼏个⽅⾯：
 # chunk_overlap 参数⽤于指定每个⽂档块之间的重叠字符数。这意味着，当⽂档被拆分成较⼩的块时，每个块的末尾部分会与下⼀个块的开头部分有⼀定数量的重叠字符。
 # 第⼀个块包含字符 1 到 1000。第⼆个块包含字符 801 到 1800。第三个块包含字符 1601 到 2600。
 chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever.invoke("猫的特征")[0]

"""
page_content='聽覺[编辑]
貓每隻⽿各有32條獨⽴的肌⾁控制⽿殼轉動，因此雙⽿可單獨朝向不同的⾳源轉動，使其向獵物移動
時仍能對周遭其他⾳源保持直接接觸。[50] 除了蘇格蘭折⽿貓這類基因突變的貓以外，貓極少有狗
常⾒的「垂⽿」，多數的貓⽿向上直⽴。當貓忿怒或受驚時，⽿朵會貼向後⽅，並發出咆哮與「嘶」
聲。
貓與⼈類對低頻聲⾳靈敏度相若。⼈類中只有極少數的調⾳師能聽到20 kHz以上的⾼頻聲⾳（8.4度
的⼋度⾳），貓則可達64kHz（10度的⼋度⾳），⽐⼈類要⾼1.6個⼋度⾳，甚⾄⽐狗要⾼1個⼋度；
但是貓辨別⾳差須開最少5度，⽐起⼈類辨別⾳差須開最少0.5度來得粗疏。[51][47]
嗅覺[编辑]
家貓的嗅覺較⼈類靈敏14倍。[52]貓的⿐腔內有2億個嗅覺受器，數量甚⾄超過某些品種的狗（狗嗅
覺細胞約1.25億～2.2億）。
味覺[编辑]
貓早期演化時由於基因突變，失去了甜的味覺，[53]但貓不光能感知酸、苦、鹹味，选择适合⾃⼰⼝
味的⻝物，还能尝出⽔的味道，这⼀点是其他动物所不及的。不过总括来说猫的味觉不算完善，相⽐
⼀般⼈類平均有9000個味蕾，貓⼀般平均僅有473個味蕾且不喜好低於室溫之⻝物。故此，貓辨認⻝
物乃憑嗅覺多於味覺。[47]
觸覺[编辑]
貓在磨蹭時身上會散發出特別的費洛蒙，當這些獨有的費洛蒙留下時，⽬的就是在宣誓主權，提醒其
它貓這是我的，其實這種⾏為算是⼀種標記地盤的象徵，會讓牠們有感到安⼼及安全感。
被⽑[编辑]
主条⽬：貓的⽑⾊遺傳和顏⾊
⻑度[编辑]
貓主要可以依據被⽑⻑度分為⻑⽑貓，短⽑貓和無⽑貓。' metadata={'source': 'https://z
h.wikipedia.org/wiki/%E7%8C%AB', 'title': '猫 - 维基百科，⾃由的百科全书', 'la
nguage': 'zh'}
"""
```
填充了我们将要进⾏Retriever的索引，我们可以轻松地将其转换为⼀个⼯具（代理程序正确使⽤所需的格式）
```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
   retriever,
   "wiki_search",
   "搜索维基百科",
)
```
使⽤语⾔模型
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model_name="gpt-4")

from langchain_core.messages import HumanMessage
response = model.invoke([HumanMessage(content="hi!")])
response.content

"""
'Hello! How can I assist you today?'
"""

model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke([HumanMessage(content="今天上海天⽓怎么样")
                                    ])
print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

"""
ContentString:
ToolCalls: [{'name': 'tavily_search_results_json', 'args': {'query': '今天上
海天⽓'}, 'id': 'call_EOxYscVIVjttlbztWoR1CvTm', 'type': 'tool_call'}]
"""
```
### 4. 创建代理程序
```python
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(model, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)
```
### 5. 运行代理
```python
print(agent_executor.invoke({"input": "你好"}))

"""
{'input': '你好', 'output': '你好！有什么我可以帮助你的吗？'}
"""
```
### 6. 添加记忆
```python
from langchain_core.messages import AIMessage, HumanMessage
agent_executor.invoke(
   {
      "chat_history": [
         HumanMessage(content="hi! my name is bob"),
         AIMessage(content="你好Bob！我今天能帮你什么？"),
      ],
      "input": "我的名字是什么?",
   }
)
```
```python
agent_with_chat_history = RunnableWithMessageHistory(
 agent_executor,
 get_session_history,
 input_messages_key="input",
 history_messages_key="chat_history",
)

response = agent_with_chat_history.invoke(
   {"input": "Hi，我的名字是Cyber"},
   config={"configurable": {"session_id": "123"}},
)
```
## 九、Embedding
embedding-models
## 十、RAG
检索增强⽣成（RAG）是指对⼤型语⾔模型输出进⾏优化，使其能够在⽣成响应之前引⽤训练数据来源之外的权威知识库。⼤型语⾔模型（LLM）⽤海量数据进⾏训练，使⽤数⼗亿个参数为回答问题、翻译语⾔和完成句⼦等任务⽣成原始输出。在 LLM 本就强⼤的功能基础上，RAG 将其扩展为能访问特定领域或组织的内部知识库，所有这些都⽆需重新训练模型。这是⼀种经济⾼效地改进 LLM 输出的⽅法，让它在各种情境下都能保持相关性、准确性和实⽤性。

### AI助手：小帽问答App(D9/rag-app)
```bash
streamlit run ./txt_search.py --server.port 1234
```


