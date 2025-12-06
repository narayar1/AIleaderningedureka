from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import os
import shutil
import re


# Load environment variables
load_dotenv()

# Configure Google AI API 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in a .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)



def extract_text_from_pdf(file):
    file_extension =  os.path.splitext(file.name)[1].lower()
    if file_extension == '.pdf':
     loader = PyPDFLoader(temp_file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(temp_file_path)
    elif file_extension == '.txt':
        loader = TextLoader(temp_file_path)
    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])
    return text
finally:
    if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

embedding_model = GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
llm = ChatGoogleGenerativeAI(model= "gemini-2.0-flash", google_api_key = GOOGLE_API_KEY)
prompt_template = PromptTemplate(
    input_variables=["job_requiements","resume_text"],
    template="You are an expert HR and recruitement Specialist. Analysse the resume below against the job requirement.Suitability score xx%"""
)

vectorstore = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)

def store_resume_analysis(resume_text, analysis,doc_id):
   documents = split_text(analysis)
   vectorstore.add_documents(documents, ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))])
   vectorstore.persist()

def main():
    st.set_page_config(page_title="Resume Screening App", layout="wide")
    st.title("Resume Screening with lcel and Vector Store")
    job_requirements = st.text_area("Enter Job Requirements",height=300)
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if st.button("Analyze") and uploaded_file and  job_requirements:
       st.download_button("Download Analysis", data=analysis, file_name="analysis.txt")   


