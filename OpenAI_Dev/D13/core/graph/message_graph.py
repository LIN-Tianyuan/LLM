# 导入 AIMessage、HumanMessage 和 ToolMessage 类，用于表示不同类型的消息
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import MessageGraph

# 创建一个新的 MessageGraph 实例
builder = MessageGraph()

# 向消息图中添加一个节点，节点名称为 "chatbot"，该节点执行的函数接收状态并返回一个包含 AI 消息的列表
# AI 消息内容为 "Hello!"，并包含一个工具调用，工具名称为 "search"，ID 为 "123"，参数为 {"query": "X"}
builder.add_node(
    "chatbot",
    lambda state: [
        AIMessage(
            content="Hello!",
            tool_calls=[{"name": "search", "id": "123", "args": {"query": "X"}}],
        )
    ],
)

# 向消息图中添加另一个节点，节点名称为 "search"，该节点执行的函数接收状态并返回一个包含工具消息的列表
# 工具消息内容为 "Searching..."，工具调用 ID 为 "123"
builder.add_node(
    "search", lambda state: [ToolMessage(content="Searching...", tool_call_id="123")]
)

# 设置消息图的入口点为 "chatbot" 节点
builder.set_entry_point("chatbot")

# 添加一条边，从 "chatbot" 节点到 "search" 节点
builder.add_edge("chatbot", "search")

# 设置消息图的结束点为 "search" 节点
builder.set_finish_point("search")

# 编译消息图并调用其 invoke 方法，传入一个包含用户消息的列表，返回包含所有消息的字典
result = builder.compile().invoke([HumanMessage(content="Hi there. Can you search for X?")])
print(result)

"""
[HumanMessage(content='Hi there. Can you search for X?', additional_kwargs={}, response_metadata={}, id='d1e76ad1-f87a-4679-9926-f65d8df49f48'), AIMessage(content='Hello!', additional_kwargs={}, response_metadata={}, id='c84dae2d-d6ba-4211-b671-c207e87fb576', tool_calls=[{'name': 'search', 'args': {'query': 'X'}, 'id': '123', 'type': 'tool_call'}]), ToolMessage(content='Searching...', id='6fba3cb0-6826-4932-8fa2-6a2374372aca', tool_call_id='123')]
"""