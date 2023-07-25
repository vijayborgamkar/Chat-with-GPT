import os
import atexit
import streamlit as st
import tempfile
from PyPDF2 import PdfMerger
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

llm_name = "gpt-3.5-turbo"
# llm_name = "gpt-4"
os.environ["OPENAI_API_KEY"] = ""

st.set_page_config(layout="wide")

def load_db(file_path, chain_type, k):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=1), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=False,
        return_generated_question=False
    )
    return qa 

chat_history = []
qa = None

# Add a dropdown menu in the sidebar
menu_selection = st.sidebar.selectbox("Menu", ["Home", "Q&A"])

# Home page content
if menu_selection == "Home":
    st.header("Welcome to Q&A App")
    st.subheader("How Q&A Works")
    st.markdown("""
    - Upload one or more PDF files

    - Optionally, merge the uploaded PDF files into a single file

    - Ask questions based on the uploaded PDF files

    - Get answers from the Q&A model

    - Chat history is displayed below the input field
    """)

# Q&A page content
elif menu_selection == "Q&A":
    files = st.sidebar.file_uploader("Upload PDF Files", type='pdf', accept_multiple_files=True)
    
    if files and len(files) > 1:
        merge_files = st.sidebar.checkbox("Merge Uploaded PDF Files")

        if merge_files:
            merger = PdfMerger()
            for file in files:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.getvalue())
                    merger.append(temp_file.name)
                    atexit.register(os.remove, temp_file.name)  # register the file to be deleted on exit
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as merged_file:
                merger.write(merged_file.name)
                atexit.register(os.remove, merged_file.name)  # register the file to be deleted on exit
            merger.close()
            qa = load_db(merged_file.name, "stuff", 4)
    elif files:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.getvalue())
                qa = load_db(temp_file.name, "stuff", 4)
                atexit.register(os.remove, temp_file.name)  # register the file to be deleted on exit

    st.header("Chat")
    query = st.text_input("Ask a question")
    if query:
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.extend([(query, result["answer"])])
        st.write(f'User: {query}')
        st.write(f'ChatBot: {result["answer"]}')

    if st.button('Clear History'):
        chat_history = []  
