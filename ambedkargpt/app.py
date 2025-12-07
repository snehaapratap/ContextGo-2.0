import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
sys.path.insert(0, str(Path(__file__).parent))
st.set_page_config(
    page_title="AmbedkarGPT",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a5f;
        --secondary-color: #3d6b99;
        --accent-color: #f39c12;
        --bg-gradient-start: #0f0c29;
        --bg-gradient-mid: #302b63;
        --bg-gradient-end: #24243e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #3d6b99 50%, #2c5282 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #e2e8f0;
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-left: 5px solid #3d6b99;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Citation box */
    .citation-box {
        background: #fff8e1;
        border-left: 4px solid #f39c12;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3a5f 0%, #2c5282 100%);
    }
    
    /* Search type buttons */
    .search-type-btn {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        transition: all 0.3s ease;
    }
    
    /* Progress animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .processing {
        animation: pulse 1.5s infinite;
    }
    
    /* Entity tags */
    .entity-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #3d6b99 0%, #2c5282 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef3 100%);
        color: #1e3a5f;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #e2e8f0;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .feature-card h4 {
        color: #1e3a5f !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: #4a5568 !important;
        font-size: 0.9rem;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Citation box - dark mode fix */
    .citation-box {
        background: #fffbeb !important;
        border-left: 4px solid #f39c12;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1a1a1a !important;
    }
    
    /* Answer box text color fix */
    .answer-box {
        color: #1a1a1a !important;
    }
    
    /* Stats card text */
    .stat-card .stat-number,
    .stat-card .stat-label {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'gpt' not in st.session_state:
        st.session_state.gpt = None
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'stats' not in st.session_state:
        st.session_state.stats = {}


def load_system():
    from src.pipeline import AmbedkarGPT
    
    if st.session_state.gpt is None:
        with st.spinner("🔄 Initializing AmbedkarGPT..."):
            st.session_state.gpt = AmbedkarGPT(config_path="config.yaml")
    
    return st.session_state.gpt


def index_document(gpt, pdf_path):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📖 Loading PDF...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        status_text.text("✂️ Creating semantic chunks (Algorithm 1)...")
        progress_bar.progress(30)
        
        stats = gpt.index_pdf(pdf_path)
        
        progress_bar.progress(70)
        status_text.text("🔗 Building knowledge graph...")
        time.sleep(0.3)
        
        progress_bar.progress(90)
        status_text.text("🏘️ Detecting communities...")
        time.sleep(0.3)
        
        progress_bar.progress(100)
        status_text.text("✅ Indexing complete!")
        time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.indexed = True
        st.session_state.stats = stats
        
        # Save index
        gpt.save_index("data/processed")
        
        return stats
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error indexing: {str(e)}")
        return None


def display_header():
    st.markdown("""
    <div class="main-header">
        <h1>📚 AmbedkarGPT</h1>
        <p>SemRAG-Based Q&A System for Dr. B.R. Ambedkar's Works</p>
    </div>
    """, unsafe_allow_html=True)


def display_stats(stats):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('num_chunks', 0)}</div>
            <div class="stat-label">Semantic Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('num_entities', 0)}</div>
            <div class="stat-label">Entities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('num_graph_nodes', 0)}</div>
            <div class="stat-label">Graph Nodes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('num_communities', 0)}</div>
            <div class="stat-label">Communities</div>
        </div>
        """, unsafe_allow_html=True)


def display_features():
    st.markdown("### 🎯 Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h4 style="color: #1e3a5f;">Semantic Chunking</h4>
            <p style="color: #4a5568;">Algorithm 1 from SEMRAG paper - cosine similarity based</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔗</div>
            <h4 style="color: #1e3a5f;">Knowledge Graph</h4>
            <p style="color: #4a5568;">Entity extraction with relationship mapping</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h4 style="color: #1e3a5f;">Hybrid Search</h4>
            <p style="color: #4a5568;">Local (Eq.4) + Global (Eq.5) retrieval</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h4 style="color: #1e3a5f;">Local LLM</h4>
            <p style="color: #4a5568;">Powered by Llama 3.2 via Ollama</p>
        </div>
        """, unsafe_allow_html=True)


def display_chat_history():
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 AmbedkarGPT:</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)


def main():
    init_session_state()
    display_header()
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        search_type = st.radio(
            "Search Method",
            ["hybrid", "local", "global"],
            index=0,
            help="Hybrid combines Local (Equation 4) and Global (Equation 5) search"
        )
        st.markdown("---")
        if st.session_state.indexed:
            st.success("✅ Document Indexed")
            if st.session_state.stats:
                st.markdown("### 📊 Statistics")
                st.json(st.session_state.stats)
        else:
            st.warning("⚠️ No document indexed")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.gpt:
                st.session_state.gpt.answer_generator.clear_history()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📖 About
        Based on **SEMRAG** research paper:
        - Semantic Chunking
        - Knowledge Graphs
        - Community Detection
        - Hybrid Retrieval
        """)
    
    gpt = load_system()
    if not st.session_state.indexed:
        if os.path.exists("data/processed/chunks.json"):
            with st.spinner("📂 Loading pre-built index..."):
                try:
                    gpt.load_index("data/processed")
                    st.session_state.indexed = True
                    full_stats = {
                        'num_chunks': len(gpt.chunks),
                        'num_entities': gpt.graph_builder.graph.number_of_nodes(),
                        'num_graph_nodes': gpt.graph_builder.graph.number_of_nodes(),
                        'num_graph_edges': gpt.graph_builder.graph.number_of_edges(),
                        'num_communities': len(gpt.community_detector.communities),
                        'num_relationships': gpt.graph_builder.graph.number_of_edges(),
                        'num_summaries': len(gpt.summarizer.community_summaries)
                    }
                    st.session_state.stats = full_stats
                    st.success("✅ Loaded existing index!")
                    st.rerun()
                except Exception as e:
                    st.warning(f"Could not load index: {e}")
        st.markdown("### 📄 Index Document")
        
        pdf_paths = [
            "data/Ambedkar_book.pdf",
            "../Ambedkar_book.pdf"
        ]
        
        pdf_found = None
        for path in pdf_paths:
            if os.path.exists(path):
                pdf_found = path
                break
        
        if pdf_found:
            st.info(f"Found PDF: `{pdf_found}`")
            
            if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
                stats = index_document(gpt, pdf_found)
                if stats:
                    st.success("🎉 Document indexed successfully!")
                    st.rerun()
        else:
            st.error("❌ Could not find Ambedkar_book.pdf. Please place it in the data/ folder.")
            
            uploaded_file = st.file_uploader("Or upload a PDF", type="pdf")
            if uploaded_file:
                os.makedirs("data", exist_ok=True)
                save_path = "data/uploaded.pdf"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("🚀 Index Uploaded PDF", type="primary"):
                    stats = index_document(gpt, save_path)
                    if stats:
                        st.success("🎉 Document indexed successfully!")
                        st.rerun()
    
    else:
        if st.session_state.stats:
            display_stats(st.session_state.stats)
        st.markdown("---")
        display_features()
        st.markdown("---")
        st.markdown("### 💬 Ask a Question")
        st.markdown("**Try these sample questions:**")
        sample_cols = st.columns(3)
        sample_questions = [
            "What is caste according to Ambedkar?",
            "What does Ambedkar say about endogamy?",
            "What solution does Ambedkar propose?"
        ]
        
        clicked_sample = None
        for i, q in enumerate(sample_questions):
            with sample_cols[i]:
                if st.button(f"📝 {q[:30]}...", key=f"sample_{i}", use_container_width=True):
                    clicked_sample = q
        question = st.text_input(
            "Your question:",
            value=clicked_sample if clicked_sample else "",
            placeholder="Ask anything about Dr. Ambedkar's works...",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        if ask_button and question:
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question
            })
            
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    result = gpt.query(question, search_type=search_type)
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result['answer']
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Conversation")
            display_chat_history()
        if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
            with st.expander("📊 Last Query Details"):
                last_q = st.session_state.chat_history[-2]['content']
                if st.session_state.gpt:
                    try:
                        result = gpt.query(last_q, search_type=search_type)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Entities Found:**")
                            if result.get('entities_used'):
                                for e in result['entities_used'][:5]:
                                    st.markdown(f"<span class='entity-tag'>{e}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**Search Type:** {result.get('search_type', 'hybrid')}")
                            st.markdown(f"**Local Entities:** {result.get('local_entities', 0)}")
                            st.markdown(f"**Communities Used:** {result.get('global_communities', 0)}")
                        
                        if result.get('citations'):
                            st.markdown("**Sources:**")
                            for cit in result['citations'][:3]:
                                st.markdown(f"""
                                <div style="background: #fffbeb; border-left: 4px solid #f39c12; padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;">
                                    <span style="color: #1a1a1a;"><strong>[{cit['id']}]</strong> {cit['excerpt'][:150]}...</span>
                                </div>
                                """, unsafe_allow_html=True)
                    except:
                        pass


if __name__ == "__main__":
    main()

