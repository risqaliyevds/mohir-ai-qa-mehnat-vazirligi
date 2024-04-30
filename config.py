from langchain import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
os.environ['BOOK_PATH'] = "source/Mehnat kodeksi en.pdf"
os.environ['BOOK_NAME'] = "mohir-ai-demo-mehnat-vazirligi"
os.environ['BOT_LOGO_PATH'] = "source/logo.jpg"
os.environ['DB_PATH'] = "D:/mata/work/mohir-ai-qa-mehnat-vazirligi/database"
