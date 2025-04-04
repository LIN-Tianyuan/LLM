"""
读取一篇博客网页（关于 AI Agent）
向量化网页内容，建立本地知识库
通过“检索工具”让 GPT 自动查询知识库回答问题
记住你和它的对话内容，比如你叫什么名字
支持多轮提问，保留上下文
"""

# 安装需要的库
# pip install --upgrade langchain langchain-community langchainhub langchain-chroma bs4 langgraph
# 导入 BeautifulSoup 库，用于解析 HTML 文档
import bs4
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从 langgraph.checkpoint.memory 模块中导入 MemorySaver 类，用于保存内存检查点
from langgraph.checkpoint.memory import MemorySaver
# 从 langgraph.prebuilt 模块中导入 create_react_agent 函数，用于创建反应代理
from langgraph.prebuilt import create_react_agent

# 创建一个 MemorySaver 对象，用于保存代理的内存状态
memory = MemorySaver()

# 创建一个 ChatOpenAI 对象，指定使用 "gpt-4" 模型，温度设为 0
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

"""
从网页抓取内容
切割成小块（分块）
嵌入到向量数据库（Chroma）
建一个可查询的知识检索器（retriever）
"""
### 构建检索器 ###
# 创建一个 WebBaseLoader 对象，用于加载指定网页的内容
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),  # 指定网页路径
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")  # 只解析指定的 HTML 类
        )
    ),
)

# 加载网页文档内容
docs = loader.load()
# 创建一个 RecursiveCharacterTextSplitter 对象，用于将文档分割成小块文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 将文档分割成小块文本
splits = text_splitter.split_documents(docs)
# 创建一个 Chroma 对象，用于将分割后的文本转换为向量存储
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()
"""
把刚才的 retriever 封装成了一个“工具”
这个工具可以被代理（Agent）调用，就像说：“我需要查知识库”
"""
### 构建检索工具 ###
# 创建一个检索工具，指定检索器、工具名称和说明
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "搜索并返回《自主代理》博客文章摘录。",
)
# 将检索工具放入工具列表中
tools = [tool]
print(tool.invoke("任务分解"))
"""
create_react_agent 创建了一个“ReAct”智能体（可以自动思考 + 工具调用）
MemorySaver 会保存对话中的上下文记忆（例如用户说过的话、名字等）
"""
# 创建一个反应代理，指定使用的语言模型、工具和内存检查点
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
# 配置代理的参数
config = {"configurable": {"thread_id": "abc123"}}
"""
第一个问题会让代理自动调用知识库工具，找到相关内容再回答你
"""
# 定义查询内容
query = "什么是任务分解？"
# 向代理发送查询，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")

"""
后面的问题会基于记忆（thread_id 是同一个）保持上下文
"""
# 定义另一个查询内容
query = "常见的做法有哪些？"
# 向代理发送查询，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")

# 向代理发送消息，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content="你好，我叫Cyber")]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")

# 定义第三个查询内容
query = "我叫什么名字？"

# 向代理发送查询，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")


"""
Fig. 1. Overview of a LLM-powered autonomous agent system.
Component One: Planning#
A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
Task Decomposition#
Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.

Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.

Fig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)
The system comprises of 4 stages:
(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.
Instruction:

(3) Task execution: Expert models execute on the specific tasks and log results.
Instruction:

With the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.
{'agent': {'messages': [AIMessage(content='任务分解是一种项目管理技术，它将大型复杂任务或项目分解为更小、更易管理的部分。这些部分被称为子任务，每个子任务都是一个可以独立完成的工作单元。任务分解的主要目标是使工作更易于理解和管理，从而提高工作效率和质量。这种方法也有助于更准确地估计任务的时间和成本，更有效地分配资源，更好地控制项目风险。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 143, 'prompt_tokens': 77, 'total_tokens': 220, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-dc39c85b-bdd1-46ee-bbed-593dd3fe2dd6-0', usage_metadata={'input_tokens': 77, 'output_tokens': 143, 'total_tokens': 220, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}
----
{'agent': {'messages': [AIMessage(content='任务分解的常见做法包括：\n\n1. **工作分解结构（WBS）**：这是一种常见的任务分解方法，它将项目分解为可管理的小部分，每个部分都有明确的目标和责任人。WBS是一个层次结构，顶层是项目目标，下面的层次是项目的各个部分或阶段，再下面的层次是具体的任务或活动。\n\n2. **敏捷方法**：在敏捷项目管理中，任务分解是将大型用户故事（User Story）分解为更小、更具体的任务的过程。这些任务可以在一个迭代或冲刺（Sprint）中完成。\n\n3. **关键路径法（CPM）**：这是一种项目管理技术，它通过确定项目的关键任务和它们之间的依赖关系，来帮助项目经理优化项目的时间表。\n\n4. **PERT图**：这是一种图形化的任务分解方法，它通过绘制任务之间的依赖关系，帮助项目经理理解任务的顺序和时间安排。\n\n5. **甘特图**：这是一种常见的项目管理工具，它可以清晰地显示任务的开始和结束日期，以及任务之间的依赖关系。\n\n以上就是一些常见的任务分解方法，不同的项目可能需要使用不同的方法，或者结合使用多种方法。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 396, 'prompt_tokens': 237, 'total_tokens': 633, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-c24ce18e-ae8c-46e4-999d-344ff4122fb8-0', usage_metadata={'input_tokens': 237, 'output_tokens': 396, 'total_tokens': 633, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}
----
{'agent': {'messages': [AIMessage(content='你好，Cyber！很高兴认识你。有什么我可以帮助你的吗？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 647, 'total_tokens': 680, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-be204eff-9bed-4983-8c2f-9383eb09e540-0', usage_metadata={'input_tokens': 647, 'output_tokens': 33, 'total_tokens': 680, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}
----
{'agent': {'messages': [AIMessage(content='你刚刚告诉我，你的名字叫Cyber。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 695, 'total_tokens': 716, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-0d051bd1-f538-4e2e-8836-c8bd628e889f-0', usage_metadata={'input_tokens': 695, 'output_tokens': 21, 'total_tokens': 716, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}
----
"""