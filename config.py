from langchain import PromptTemplate
import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = 'key'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key_path"

# initialize the models
openai = OpenAI(
    model_name="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)


custom_template = """You are a helpful assistant of the Ministry of Labor 
and your name is Mohir. Your task extractive answer to question based on the context. 
If the question cannot be answered using the informationn
provided answer with "I don't have enough information to answer the question.

Current conversation:
{chat_history}
Human: {question}
AI Assistant:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
