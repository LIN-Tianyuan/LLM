# 导入所需的模块和类
import operator
from contextlib import contextmanager
from typing import Annotated, Sequence

import httpx
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.channels.context import Context
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt import ToolNode

# 初始化一个ChatOpenAI模型实例，设置温度为0
model = ChatOpenAI(temperature=0)


# 定义一个代理上下文类，继承自BaseModel
class AgentContext(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # 定义一个httpx客户端会话属性
    httpx_session: httpx.Client


# 创建一个上下文管理器，用于创建和管理AgentContext实例
@contextmanager
def make_agent_context(config: RunnableConfig):
    # 创建一个httpx客户端会话
    session = httpx.Client()
    try:
        # 生成一个包含httpx会话的AgentContext实例
        yield AgentContext(httpx_session=session)
    finally:
        # 关闭httpx会话
        session.close()


# 定义代理状态类，包含消息序列和上下文
class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: Annotated[AgentContext, Context(make_agent_context)]


# 定义一个工具函数，用于搜索查询
@tool
def search(query: str):
    """Call to surf the web."""
    # 这是实际实现的占位符
    # 不要让LLM知道这个😊
    return ["The answer to your question lies within."]


# 创建一个包含搜索工具的工具执行器
tools = [search]
tool_executor = ToolNode(tools)


# 定义一个函数，用于确定是否继续
def should_continue(state):
    messages = state.messages
    last_message = messages[-1]
    # 如果没有函数调用，则结束
    if not last_message.tool_calls:
        return "end"
    # 否则继续
    else:
        return "continue"


# 定义一个函数，用于调用模型
def call_model(state):
    # 使用上下文值
    req = state.context.httpx_session.get("https://www.langchain.com/")
    assert req.status_code == 200, req
    messages = state.messages
    response = model.invoke(messages)
    # 返回一个列表，因为这将被添加到现有列表中
    return {"messages": [response]}


# 定义一个函数，用于执行工具
def call_tool(state):
    messages = state.messages
    # 根据继续条件
    # 我们知道最后一条消息涉及函数调用
    last_message = messages[-1]
    # 我们从函数调用中构建一个ToolInvocation
    tool_call = last_message.tool_calls[0]
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # 我们调用工具执行器并返回响应
    response = tool_executor.invoke(action)
    # 我们使用响应创建一个ToolMessage
    tool_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # 返回一个列表，因为这将被添加到现有列表中
    return {"messages": [tool_message]}


# 定义一个新的状态图
workflow = StateGraph(AgentState)

# 定义我们将在其之间循环的两个节点
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# 将入口点设置为`agent`
# 这意味着这是第一个被调用的节点
workflow.add_edge(START, "agent")

# 我们现在添加一个条件边
workflow.add_conditional_edges(
    # 首先，我们定义起始节点。我们使用`agent`。
    # 这意味着这些边是在调用`agent`节点之后采取的。
    "agent",
    # 接下来，我们传入确定下一个被调用节点的函数。
    should_continue,
    # 最后我们传入一个映射。
    # 键是字符串，值是其他节点。
    # END是一个特殊节点，表示图应该结束。
    # 将调用`should_continue`，然后其输出将与此映射中的键匹配。
    # 根据匹配的结果，调用相应的节点。
    {
        # 如果是`action`，则调用工具节点。
        "continue": "action",
        # 否则结束。
        "end": END,
    },
)

# 我们现在从`tools`到`agent`添加一个正常边。
# 这意味着在调用`tools`之后，调用`agent`节点。
workflow.add_edge("action", "agent")

# 最后，我们编译它！
# 这将其编译为LangChain Runnable，
# 这意味着你可以像使用其他runnable一样使用它
app = workflow.compile()

# 创建一些初始消息
initial_messages = [
    HumanMessage(content="langchain最新版本")
]

# 创建初始状态
initial_state = AgentState(
    messages=initial_messages,
    context=AgentContext(httpx_session=httpx.Client())
)

# 调用app.invoke，并传递初始状态
result = app.invoke(initial_state)
print(result)

# 假设 app 是你的工作流应用实例
graph_png = app.get_graph().draw_mermaid_png()
# 将生成的图片保存到文件
with open("context_reducer.png", "wb") as f:
    f.write(graph_png)
