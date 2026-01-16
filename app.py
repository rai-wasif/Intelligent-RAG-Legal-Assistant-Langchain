import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

# ---------------------------------------------------------
# 🔑 CONFIGURATION
# ---------------------------------------------------------
def get_hf_token():
    """Securely retrieve HuggingFace token"""
    try:
        return st.secrets["HF_TOKEN"]
    except:
        return os.getenv("HF_TOKEN", "123wrong")

HF_TOKEN = get_hf_token()
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# ---------------------------------------------------------
# 🧠 LOAD RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    """Load embeddings, vector database, and LLM"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(
            "vectorstore/db_faiss", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        llm = HuggingFaceEndpoint(
            repo_id=REPO_ID,
            task="text-generation",
            max_new_tokens=300,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            huggingfacehub_api_token=HF_TOKEN
        )
        
        return db, llm, None
    except Exception as e:
        return None, None, str(e)

# ---------------------------------------------------------
# 🤖 RAG PIPELINE  
# ---------------------------------------------------------
def format_constitutional_answer(docs, query):
    """Format answer from retrieved documents without LLM"""
    if not docs:
        return "No relevant constitutional provisions found for your question.", [], None
    
    # Extract article numbers from content
    answer_parts = ["**Based on the Constitution of Pakistan:**\n"]
    
    for i, doc in enumerate(docs[:2], 1):  # Use top 2 most relevant
        content = doc.page_content.strip()
        
        # Try to extract article number
        if "Article" in content or "Clause" in content:
            answer_parts.append(f"\n📜 **Reference {i}:**")
        
        # Clean and format the content
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Skip very short lines
                answer_parts.append(f"{line}")
    
    answer_parts.append(f"\n\n💡 *The above provisions directly address your question about: \"{query}\"*")
    
    return "\n".join(answer_parts), [doc.page_content for doc in docs], None

def get_constitutional_answer(query, db, llm):
    """Retrieve and format constitutional answer"""
    try:
        # Search for relevant documents
        docs = db.similarity_search(query, k=3)
        
        if not docs:
            return "I couldn't find relevant constitutional provisions for your question. Try rephrasing or asking about specific Articles.", [], None
        
        # Build context for LLM
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Optimized prompt for Mistral
        prompt = f"""<s>[INST] You are a constitutional law expert. Answer the question based ONLY on the provided constitutional text. Be concise and cite article numbers.

Constitutional Text:
{context}

Question: {query}

Provide a clear 2-3 sentence answer. [/INST]"""

        try:
            # Try LLM generation
            response = llm.invoke(prompt)
            
            # Clean Mistral response
            answer = str(response).strip()
            
            # Remove instruction tags if present
            if "[/INST]" in answer:
                answer = answer.split("[/INST]")[-1].strip()
            if "<s>" in answer:
                answer = answer.replace("<s>", "").strip()
            if "[INST]" in answer:
                answer = answer.split("[INST]")[0].strip()
            
            # Validate answer quality
            if len(answer) > 50 and "constitution" in answer.lower():
                # LLM generated good answer
                return answer, [doc.page_content for doc in docs], None
            else:
                # LLM failed, use formatted fallback
                return format_constitutional_answer(docs, query)
                
        except Exception as llm_error:
            # LLM failed, use formatted fallback
            return format_constitutional_answer(docs, query)
            
    except Exception as e:
        return None, [], str(e)

# ---------------------------------------------------------
# 🎨 UI SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Constitution of Pakistan AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stat-box {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">⚖️ Constitution of Pakistan AI</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.9;">Ask questions about Pakistani Constitutional Law • Powered by AI</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 📊 INITIALIZATION
# ---------------------------------------------------------
if 'resources_loaded' not in st.session_state:
    with st.spinner("⚙️ Loading AI models and constitutional database..."):
        db, llm, error = load_resources()
        if error:
            st.error(f"❌ Initialization failed: {error}")
            st.info("💡 Ensure: vectorstore/db_faiss exists and HF token is valid")
            st.stop()
        st.session_state.db = db
        st.session_state.llm = llm
        st.session_state.resources_loaded = True

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """Hello! I'm your AI assistant for the Constitution of Pakistan. Ask me about:

• **Specific Articles** (e.g., "What does Article 6 say about High Treason?")
• **Fundamental Rights**
• **Presidential qualifications**
• **Arrest and detention rights**
• **National Assembly and Senate**

What would you like to know?"""
    }]

# ---------------------------------------------------------
# 💬 CHAT INTERFACE
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View Constitutional References"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Reference {i}:**")
                    st.text(source[:400] + "..." if len(source) > 400 else source)
                    if i < len(msg["sources"]):
                        st.divider()

# User input
prompt = st.chat_input("💭 Ask about any constitutional provision...")

if prompt:
    # Validate
    if len(prompt.strip()) < 3:
        st.warning("⚠️ Please enter a more detailed question.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching constitutional provisions..."):
            answer, sources, error = get_constitutional_answer(
                prompt,
                st.session_state.db,
                st.session_state.llm
            )
            
            if error and not answer:
                st.error(f"❌ Error: {error}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {error}"
                })
            elif answer:
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources if sources else []
                })
            else:
                st.warning("⚠️ Unable to generate response. Please try rephrasing.")

# ---------------------------------------------------------
# 📌 SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💡 Example Questions")
    examples = [
        "What does Article 6 say about High Treason?",
        "What are fundamental rights?",
        "Can a person be arrested without being informed?",
        "What are Presidential qualifications?",
        "Explain Article 10"
    ]
    for ex in examples:
        st.markdown(f"• {ex}")
    
    st.divider()
    
    st.markdown("### 📊 Session Stats")
    user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"""
    <div class="stat-box">
        <h3 style="margin:0;">Questions Asked</h3>
        <h2 style="margin:0.5rem 0 0 0; color:#3b82f6;">{user_msgs}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🔧 System Info")
    st.caption(f"Model: {REPO_ID.split('/')[-1]}")
    st.caption("Vector DB: ✅ Loaded")
    st.caption("LLM: ✅ Ready")
    
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Footer
st.divider()
cols = st.columns(3)
with cols[0]:
    st.caption("💡 **Tip:** Be specific for better answers")
with cols[1]:
    st.caption("⚖️ **Source:** Constitution of Pakistan 1973")
with cols[2]:
    st.caption(f"🤖 **Powered by:** {REPO_ID.split('/')[-1]} + LangChain")