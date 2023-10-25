from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os

def initializationPinocone():
    pinecone.init(
        api_key=os.environ['PINOCONE_KEY'],
        environment=os.environ['PINOCONE_ENV']
    )
    # return pinecone.Index(index_name)

def getEmbeddings():
    return OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

def getExistsDocs(index_name, embeddings):
    return Pinecone.from_existing_index(index_name, embeddings)

def createDocs(index_name, doc_path, embeddings):
    pinecone.create_index(index_name, dimension=1536)
    loader = PyPDFLoader(doc_path)
    pages = loader.load_and_split()
    return Pinecone.from_documents(pages, embeddings, index_name=index_name)

def getRetriever(index_name):
    initializationPinocone()
    embeddings = getEmbeddings()

    if index_name in pinecone.list_indexes():
        docsearch = getExistsDocs(index_name, embeddings)
    else:
        docsearch = createDocs(index_name, os.environ['BOOK_PATH'], embeddings)

    retriever = docsearch.as_retriever(
            search_type="similarity", search_kwargs={"k": 5})

    return retriever

retrieverQA = getRetriever(os.environ['BOOK_NAME'])