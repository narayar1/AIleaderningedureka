from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever





import os   
from dotenv import load_dotenv

load_dotenv()
# Configure Google AI API 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = WikipediaLoader(query= "MKUltra")
documents = loader.load()
print(f" Length of documents beore slitting is : {len(documents)}")
# print(len(documents))
# print(documents)
text_splitter = CharacterTextSplitter(chunk_size=500)
docs = text_splitter.split_documents(documents)
print(f" Length of documents after splitting is :{len(docs)}")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="chroma_store/wikipedia")
vectorstore.persist()

question  = "When was this declassified"

llm = ChatGoogleGenerativeAI(
     model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
       temperature=0
    )
    
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

import logging
logging.basicConfig(level=logging.INFO) 
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(question)
print(unique_docs)


