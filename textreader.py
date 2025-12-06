from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
import os
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import google.generativeai as genai

#Load environment variables
load_dotenv()

print(os.getcwd())
# Configure Google AI API 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def us_constituion_helper(question):
    """
    Helper function to answer questions about the US Constitution.
    Uses a retriever to find relevant documents and a language model to generate answers.
    """
    #Load the file
    # loader = TextLoader("F:\\agenticai\\edureka\\agenticai\\projects\\python3.13\\us_constitution.txt", encoding="utf-8")  
    loader = TextLoader("us.txt", encoding="utf-8")  
    
    #Split it into chunks
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    #Create a vector store
    VECTOR_STORE_DIR = "chroma_store"
    vectorstore = Chroma.from_documents(split_docs, embedding_model, persist_directory="chroma_store/textreader")
    vectorstore.persist()

    vector_new_connection = Chroma(persist_directory="chroma_store/textreader", embedding_function=embedding_model)
    new_doc = "What is the Eleventh Amendment US Constitution?"
    similar_docs = vector_new_connection.similarity_search(new_doc,k=4)
    print(similar_docs[0].page_content) # Print the first 500 characters for brevity

    # #Use chatopenai to create a retriever
    # llm = ChatGoogleGenerativeAI(
    #             model="gemini-2.0-flash",
    #             google_api_key=GOOGLE_API_KEY,
    #             temperature=0
    #         )
    # compressor = LLMChainExtractor.from_llm(llm)
    # # Print the first 500 characters for brevity

us_constituion_helper("What is the purpose of the US Constitution?")




