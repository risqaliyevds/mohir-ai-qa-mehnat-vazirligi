from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

os.environ['BOOK_PATH'] = 'source/Mehnat kodeksi en.pdf'


class QASystem:
    def __init__(self, doc_name) -> None:
        self.doc_name = doc_name
        self.db = None

    def loader(self):
        loader = PyPDFLoader(self.doc_name)
        documents = loader.load()
        return documents

    def splitter(self):
        documents = self.loader()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    def createEmbeddings(self):
        embeddings = OpenAIEmbeddings()
        return embeddings

    def checkDb(self):
        if not self.db:
            self.texts = self.splitter()
            self.embeddings = self.createEmbeddings()
            self.db = Chroma.from_documents(self.texts, self.embeddings)

    def retriever(self):
        self.checkDb()
        retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": 2})
        return retriever


retriver = QASystem(os.environ['BOOK_PATH']).retriever()
