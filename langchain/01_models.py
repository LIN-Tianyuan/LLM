import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI_Dev: ")

model = init_chat_model("gpt-4o-mini", model_provider="openai")

print(model.invoke('帮我讲个笑话吧！'))