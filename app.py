# RAG Q&A conversation with PDF including chat history
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFium2Loader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')

# Creating embedding
embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

# Heading of the application
st.title("Conversational RAG with user input PDF and chat history")
st.write("Upload PDF and chat with content")
# st.secrets['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Asking user to share groq api key
groq_api_key = st.text_input("Enter your Groq API key:", type="password")

# Check if groq api key is available
if groq_api_key:
    # Creating model
    llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)
    
    # Chat interface
    session_id = st.text_input("Session ID", value="default_session")

    # successfully manage the history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Uploading pdf file
    uploaded_files = st.file_uploader("Upload PDF file", 
                                     type=['pdf'], 
                                     accept_multiple_files=True)

    # Process uploaded file
    if uploaded_files:
        documents = []

        save_dir = "Projects/RAG Q&A conversation"
        os.makedirs(save_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            temp_pdf = os.path.join(save_dir, uploaded_file.name)
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                # f.write(uploaded_file)
                # f.name=uploaded_file.name
        
            # loader = PyPDFLoader(temp_pdf)
            loader = PyPDFium2Loader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embedding of documents to store in DB and create retriever
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vector_store.as_retriever()
        
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, 
                                                                 retriever, 
                                                                 contextualize_q_prompt)
        
        # answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {'input':user_input},
                config={
                    "configurable":{'session_id':session_id}
                    },
            ) 

            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)

else:

    st.warning("Please enter your groq api key.")

