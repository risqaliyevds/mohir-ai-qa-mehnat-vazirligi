from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import pinecone
import glob
import os


def getEmbeddings():
    return OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

def getExistsDocs(name_of_db, embeddings):
    return FAISS.load_local(os.environ["DB_PATH"] + "/" + name_of_db, embeddings)

def createDocs(name_of_db, doc_path, embeddings):
    loader = PyPDFLoader(doc_path)
    documents = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    save_path = os.environ['DB_PATH'] + "/" + name_of_db
    db.save_local(save_path)

def getRetriever(name_of_db):
    embeddings = getEmbeddings()
    folders = glob.glob(os.environ['DB_PATH'] + "/*")
    db_folders = [str(path).split("\\")[-1] for path in folders]

    if name_of_db in db_folders:
        docsearch = getExistsDocs(name_of_db, embeddings)
        print("Database exists")
    else:
        createDocs(name_of_db, os.environ['BOOK_PATH'], embeddings)
        docsearch = getExistsDocs(name_of_db, embeddings)
        print("Database created")

    retriever = docsearch.as_retriever(
            search_type="similarity", search_kwargs={"k": 5})

    return retriever

retrieverQA = getRetriever(os.environ['BOOK_NAME'])
