from langchain import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
os.environ['BOOK_PATH'] = "source/Mehnat kodeksi en.pdf"
os.environ['PINOCONE_KEY'] = "ab45d421-7ce0-46cf-9bdf-f4aad2da09de"
os.environ['PINOCONE_ENV'] = "gcp-starter"
os.environ['BOOK_NAME'] = "mohir-ai-demo-mehnat-vazirligi"
os.environ['BOT_LOGO_PATH'] = "source/logo.jpg"
