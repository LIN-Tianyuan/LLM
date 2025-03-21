# 导入所需的类型提示和类
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# 在运⾏时传递回调函数

# 定义一个日志处理器类，继承自BaseCallbackHandler
class LoggingHandler(BaseCallbackHandler):
    # 当聊天模型开始时调用的方法
    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")  # 打印“Chat model started”

    # 当LLM结束时调用的方法
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")  # 打印“Chat model ended, response: {response}”

    # 当链开始时调用的方法
    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain {serialized.get('name')} started")  # 打印“Chain {serialized.get('name')} started”

    # 当链结束时调用的方法
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")  # 打印“Chain ended, outputs: {outputs}”


# 创建一个包含LoggingHandler实例的回调列表
callbacks = [LoggingHandler()]

# 实例化一个ChatOpenAI对象，使用gpt-4模型
llm = ChatOpenAI(model_name="gpt-4")

# 创建一个聊天提示模板，模板内容为“What is 1 + {number}?”
prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

# 将提示模板和LLM组合成一个链
chain = prompt | llm

# 调用链的invoke方法，传入参数number为"2"，并配置回调
chain.invoke({"number": "2"}, config={"callbacks": callbacks})

"""
Chain ChatPromptTemplate started
Chain ended, outputs: messages=[HumanMessage(content='What is 1 + 2?', additional_kwargs={}, response_metadata={})]
Chat model started
Chat model ended, response: generations=[[ChatGeneration(text='3', generation_info={'finish_reason': 'stop', 'logprobs': None}, message=AIMessage(content='3', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 2, 'prompt_tokens': 15, 'total_tokens': 17, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-334c2896-0ce9-4644-b6aa-d27e01f808c3-0', usage_metadata={'input_tokens': 15, 'output_tokens': 2, 'total_tokens': 17, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}))]] llm_output={'token_usage': {'completion_tokens': 2, 'prompt_tokens': 15, 'total_tokens': 17, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None} run=None type='LLMResult'
Chain ended, outputs: content='3' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 2, 'prompt_tokens': 15, 'total_tokens': 17, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4-0613', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-334c2896-0ce9-4644-b6aa-d27e01f808c3-0' usage_metadata={'input_tokens': 15, 'output_tokens': 2, 'total_tokens': 17, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
"""
