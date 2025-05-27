from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chat_models import init_chat_model
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = init_chat_model("gpt-3.5-turbo-0125", model_provider="openai")

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is Carl."),
    ("human", "{user_input}"),
])

prompt = prompt_template.invoke("Hello, there!")

response = model.invoke(prompt)

print(response)