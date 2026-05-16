import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
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

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(api_key, model):
    return ChatGroq(groq_api_key=api_key, model_name=model)

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
        
        embeddings = get_embeddings()
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
            llm = get_llm(groq_api_key, model_name)
            
            # 1. Tools
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            retriever_tool = create_retriever_tool(
                retriever,
                "documentsearch",
                "Use this to find info from the uploaded document. This is your primary source."
            )
            
            search_tool = DuckDuckGoSearchRun()
            tools = [retriever_tool, search_tool]
            
            # 2. ReAct Agent Prompt
            # We'll use a standard ReAct prompt from LangChain Hub or define it
            prompt = hub.pull("hwchase17/react-chat")
            
            # 3. Agent
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            
            # Convert session state history to LangChain messages
            history = ""
            for msg in st.session_state.chat_history[:-1]:
                role = "Human" if msg["role"] == "user" else "AI"
                history += f"{role}: {msg['content']}\n"
            
            # 4. Invoke
            try:
                response = agent_executor.invoke({
                    "input": latest_query,
                    "chat_history": history,
                    "tools": [t.name for t in tools],
                    "tool_names": ", ".join([t.name for t in tools])
                })
                output = response['output']
            except Exception as e:
                output = f"I encountered an error while processing: {str(e)}. Please try rephrasing your question."
            
            st.session_state.chat_history.append({"role": "assistant", "content": output})
            st.rerun()

else:
    if not uploaded_file:
        st.warning("Please upload a PDF document to start chatting.")
    
st.markdown("---")
