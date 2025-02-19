import streamlit as st
import os
import shutil
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load the Groq And OpenAI Api Key
# os.environ['OPEN_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


st.title("Objectbox VectorstoreDB With Llama3 Demo")

model_name_groq = "llama3-8b-8192"  # Example: 8B model
# model_name_groq = "llama3.2-3b-preview"
model_name = "llama3.2:latest"

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name=model_name_groq)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """

)

if "vectors" not in st.session_state:
    st.session_state.vectors = None
# elif st.session_state.vectors:
#     st.session_state.vectors._db.close()  # Properly close the existing store
#     print("----------->VECTOR CLOSING NOW")
#     st.session_state.vectors = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if os.path.exists("objectbox"):
    shutil.rmtree("objectbox")

# File uploader to allow users to upload PDFs
uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)


## Vector Enbedding and Objectbox Vectorstore db

def vector_embedding():
    print("vector_embedding==================>")
    if st.session_state:
        for k,v in st.session_state.items():
            print(k, '--->', v)
    if st.session_state.vectors:
        st.session_state.vectors._db.close()  # Properly close the existing store
        st.session_state.vectors = None
        # return
    # Save uploaded PDFs to a temporary directory
    temp_dir = "./uploaded_docs"
    if os.path.exists(temp_dir):
        if os.path.exists("objectbox"):
            shutil.rmtree("objectbox")
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for uploaded_file in st.session_state.uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    st.session_state.embeddings=OllamaEmbeddings(model = model_name)
    st.session_state.loader=PyPDFDirectoryLoader(temp_dir) ## Data Ingestion
    # print("hiii",st.session_state.loader)
    st.session_state.docs=st.session_state.loader.load() ## Documents Loading
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=768)



if st.button("Process and Embed Documents"):
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.write(f"Processing {len(uploaded_files)} documents...")
    vector_embedding()
        


input_prompt=st.text_input("Enter Your Question From Documents")



# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("ObjectBox Database is ready")

import time

if input_prompt:
    if not uploaded_files:
        st.warning("Please upload files and retry...")
    elif "vectors" not in st.session_state or st.session_state.vectors is None:
        print("---------------------->", st.session_state.vectors)
        st.warning("Embedding not found! Initializing now...")
        st.session_state.uploaded_files = uploaded_files
        st.write(f"Processing {len(uploaded_files)} documents...")
        vector_embedding()
        # else:

        
    if st.session_state.vectors:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        start=time.process_time()

        response=retrieval_chain.invoke({'input':input_prompt})

        print("Response time :",time.process_time()-start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")











