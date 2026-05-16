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
from langchain_classic.chains import create_retrieval_chain
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Antigravity RAG | AI Document Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
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

# Sidebar
with st.sidebar:
    st.title("🚀 Configuration")
    st.markdown("---")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Groq API Key not found in .env")
    
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Choose Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=1
    )
    
    st.markdown("---")
    st.markdown("### How to use:")
    st.info("1. Upload a PDF document\n2. Wait for processing\n3. Ask questions in the chat")

# Header
st.title("🧠 Antigravity RAG Pipeline")
st.markdown("#### *Analyze your documents with state-of-the-art AI*")
st.markdown("---")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File Upload Section
uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("🚀 Processing document... This might take a moment."):
        # Save uploaded file temporarily
        temp_file_path = "temp_uploaded_doc.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and split document
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(final_documents, embeddings)
        
        # Cleanup temp file
        os.remove(temp_file_path)
        
        st.success("✅ Document processed successfully!")

# Chat Interface
if st.session_state.vectorstore is not None:
    st.markdown("### 💬 Chat with your Document")
    
    # Display chat history
    for message in st.session_state.chat_history:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        role_name = "👤 You" if message["role"] == "user" else "🤖 Assistant"
        
        st.markdown(f"""
        <div class="chat-message {role_class}">
            <div class="message-header">{role_name}</div>
            <div>{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

    # User input
    user_query = st.chat_input("Ask a question about your document...")

    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Re-render history with new message immediately
        st.rerun()

    # Process latest user message if any
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        latest_query = st.session_state.chat_history[-1]["content"]
        
        with st.spinner("🤖 Thinking..."):
            # Initialize LLM
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context.
            Think step by step before providing a detailed answer.
            If the answer is not in the context, say that you don't have enough information.
            
            <context>
            {context}
            </context>
            
            Question: {input}
            """)
            
            # Create chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Get response
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": latest_query})
            response_time = time.process_time() - start_time
            
            # Add assistant response to history
            answer = response['answer']
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

else:
    if not uploaded_file:
        st.warning("Please upload a PDF document to start chatting.")
    
# Footer
st.markdown("---")
