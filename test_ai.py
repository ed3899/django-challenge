from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chat_models import init_chat_model
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = init_chat_model("gpt-3.5-turbo-0125", model_provider="openai")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

response = model.invoke(prompt)

print(response)