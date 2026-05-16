import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import time


load_dotenv()

st.set_page_config(
    page_title="SEO Answering | AI Document Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        color: #1f2937;
    }
    .stApp {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    [data-testid="stSidebar"] {
        background-color: #f3f4f6;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #1f2937;
        border: 1px solid #d1d5db;
        border-radius: 8px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        border: 1px solid #e5e7eb;
    }
    .user-message {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
    }
    .assistant-message {
        background-color: #f0fdf4;
        border-left: 5px solid #10b981;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #6b7280;
    }
    h1, h2, h3 {
        color: #1e40af !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🚀 SEO Answering")
    st.markdown("---")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API Key not found in .env")
    
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Choose Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### How to use:")
    st.info("1. Upload a PDF document\n2. Wait for processing\n3. Ask questions in the chat")

st.title("🧠 SEO Answering Pipeline")
st.markdown("#### *Analyze your documents with state-of-the-art AI*")
st.markdown("---")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("🚀 Processing document... This might take a moment."):
        temp_file_path = "temp_uploaded_doc.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(final_documents, embeddings)
        
        os.remove(temp_file_path)
        
        st.success("✅ Document processed successfully!")

if st.session_state.vectorstore is not None:
    st.markdown("### 💬 Chat with your Document")
    
    for message in st.session_state.chat_history:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        role_name = "👤 You" if message["role"] == "user" else "🤖 Assistant"
        
        st.markdown(f"""
        <div class="chat-message {role_class}">
            <div class="message-header">{role_name}</div>
            <div>{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

    user_query = st.chat_input("Ask a question about your document...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        st.rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        latest_query = st.session_state.chat_history[-1]["content"]
        
        with st.spinner("🤖 Thinking..."):
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
            
            # 1. History-aware retriever
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                ]
            )
            
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            # 2. QA chain
            system_prompt = (
                "You are a highly accurate technical assistant. Use the provided context to answer the question. "
                "If the answer involves code or technical steps (like ASP.NET or CGI Perl), provide them clearly. "
                "If the question mentions marks, provide a detailed and complete answer. "
                "Format your answer cleanly using bullet points or numbered lists where appropriate. "
                "If you don't know the answer based on the context, say that the information is not available.\n\n"
                "{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                ]
            )
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # Convert session state history to LangChain messages
            history = []
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))
            
            response = rag_chain.invoke({"input": latest_query, "chat_history": history})
            
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
            st.rerun()

else:
    if not uploaded_file:
        st.warning("Please upload a PDF document to start chatting.")
    
st.markdown("---")
