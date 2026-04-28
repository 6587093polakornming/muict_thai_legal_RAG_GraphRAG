"""
Thai Legal RAG — Streamlit Chatbot
วาง file นี้ไว้ที่ root ของ repo (ระดับเดียวกับ src/)
streamlit run app.py --server.port 8080
"""
from __future__ import annotations
import os
import streamlit as st
from langchain_openai import ChatOpenAI

# Import จาก src/ (เหมือนกับ hybridrag_query_cli.py เดิม)
from src.rag.config import RAGConfig
from src.rag.hybridrag_langhchain import ThaiLegalRAG

# ──────────────────────────────────────────────
# 0. Page config  (ต้องเป็น st call แรกสุดเสมอ)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Thai Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 1. Custom CSS — warm legal theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Sarabun', sans-serif;
}

/* ── App background ── */
.stApp {
    background-color: #FAF6F1;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #3D2B1F;
}
[data-testid="stSidebar"] * {
    color: #F5ECD7 !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background-color: #5C3D2A !important;
    border: 1px solid #8B6347 !important;
    color: #F5ECD7 !important;
    border-radius: 8px;
}
[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #C4A882 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #5C3D2A !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #3D2B1F 0%, #6B4226 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(61,43,31,0.15);
}
.app-header h1 {
    font-family: 'Playfair Display', serif;
    color: #F5ECD7;
    margin: 0;
    font-size: 1.8rem;
    letter-spacing: 0.5px;
}
.app-header p {
    color: #C4A882;
    margin: 0;
    font-size: 0.85rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 0.75rem !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
div[class*="user"] .stMarkdown {
    background-color: #6B4226;
    color: #F5ECD7;
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.1rem;
    max-width: 75%;
    margin-left: auto;
    box-shadow: 0 2px 8px rgba(61,43,31,0.15);
}

/* Assistant bubble */
[data-testid="stChatMessage"]:not([data-testid*="user"]) .stMarkdown {
    background-color: #FFFFFF;
    border: 1px solid #E8DDD0;
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.1rem;
    max-width: 85%;
    box-shadow: 0 2px 8px rgba(61,43,31,0.08);
    color: #2C1810;
}

/* ── Chat input box ── */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 2px solid #C4A882 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 12px rgba(61,43,31,0.10) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6B4226 !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Sarabun', sans-serif !important;
    color: #2C1810 !important;
}

/* ── Sidebar label style ── */
.sidebar-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #C4A882;
    margin-bottom: 0.3rem;
    display: block;
}

/* ── Status badge ── */
.status-ok {
    background: #2D6A4F;
    color: #D8F3DC;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.status-warn {
    background: #7B4F00;
    color: #FFE69C;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Source card ── */
.source-card {
    background: #FDF5EC;
    border-left: 3px solid #6B4226;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    margin-top: 0.4rem;
    font-size: 0.82rem;
    color: #5C3D2A;
}
.source-card b { color: #3D2B1F; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. Sidebar — API Key + Config
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Thai Legal AI")
    st.markdown("ผู้ช่วยด้านกฎหมายแพ่งและพาณิชย์")
    st.divider()

    # --- Typhoon API Key ---
    st.markdown('<span class="sidebar-label">🔑 Typhoon API Key</span>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        label="api_key",
        type="password",
        placeholder="sk-...",
        label_visibility="collapsed",
        key="typhoon_api_key",
    )

    st.divider()

    # --- Clear chat ---
    if st.button("🗑️ ล้างประวัติการสนทนา", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag = None
        st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem;color:#8B6347;text-align:center;">'
        'Powered by Typhoon + BGE-M3<br>Senior Project MUICT</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 3. Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div style="font-size:2.5rem">⚖️</div>
    <div>
        <h1>Thai Legal AI Assistant</h1>
        <p>ผู้ช่วยด้านกฎหมายแพ่งและพาณิชย์ไทย · Hybrid RAG + BGE-M3</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 4. Session State init
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources?}]

if "rag" not in st.session_state:
    st.session_state.rag = None      # ThaiLegalRAG instance (cached)

if "last_api_key" not in st.session_state:
    st.session_state.last_api_key = ""

RETRIEVAL_LIMIT = 3
RERANKING_LIMIT = 3


# ──────────────────────────────────────────────
# 5. Build / Rebuild RAG (cached ตาม API key + limits)
# ──────────────────────────────────────────────
def build_rag(api_key: str, r_limit: int, rr_limit: int) -> ThaiLegalRAG:
    """
    โหลด embed model + reranker + qdrant client ครั้งเดียว
    ใช้ @st.cache_resource เพื่อไม่ให้โหลดซ้ำทุก rerun
    """
    llm = ChatOpenAI(
        model_name="typhoon-v2.5-30b-a3b-instruct",
        openai_api_key=api_key,
        openai_api_base="https://api.opentyphoon.ai/v1",
        temperature=0,
        max_tokens=8192,
    )
    config = RAGConfig(
        retrieval_limit=r_limit,
        reranking_limit=rr_limit,
    )
    return ThaiLegalRAG(llm=llm, config=config)


@st.cache_resource(show_spinner="⏳ กำลังโหลด BGE-M3 + Reranker (ครั้งแรกอาจนาน 1-2 นาที)...")
def get_cached_rag(api_key: str, r_limit: int, rr_limit: int) -> ThaiLegalRAG:
    """
    st.cache_resource → โหลด model ครั้งเดียวตลอด session
    key คือ tuple (api_key, r_limit, rr_limit)
    """
    return build_rag(api_key, r_limit, rr_limit)


# ──────────────────────────────────────────────
# 6. Guard — ถ้ายังไม่มี API key
# ──────────────────────────────────────────────
if not api_key_input:
    st.info("👈 กรุณาใส่ **Typhoon API Key** ในแถบด้านซ้ายก่อนเริ่มใช้งาน", icon="🔑")
    st.stop()

# โหลด RAG เมื่อมี key (cache ไว้ ไม่โหลดซ้ำ)
rag: ThaiLegalRAG = get_cached_rag(api_key_input, RETRIEVAL_LIMIT, RERANKING_LIMIT)


# ──────────────────────────────────────────────
# 7. Render chat history
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"])

        # แสดง sources ถ้ามี (เฉพาะ assistant)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 แหล่งอ้างอิงที่ใช้ตอบ", expanded=False):
                for doc in msg["sources"]:
                    meta = doc.metadata
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>Rank [{meta.get("rank", "-")}]</b> · '
                        f'Score: {meta.get("score", 0):.4f}<br>'
                        f'📜 {meta.get("law_name", "-")} มาตรา {meta.get("section_num", "-")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ──────────────────────────────────────────────
# 8. Chat input + RAG call
# ──────────────────────────────────────────────
if prompt := st.chat_input("พิมพ์คำถามด้านกฎหมายของคุณที่นี่..."):

    # -- บันทึก user message --
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # -- เรียก RAG --
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("🔍 กำลังค้นหาและวิเคราะห์กฎหมาย..."):
            try:
                answer, docs = rag.chat(prompt)
            except Exception as e:
                answer = f"❌ เกิดข้อผิดพลาด: {e}"
                docs = []

        st.markdown(answer)

        # แสดง sources
        if docs:
            with st.expander("📚 แหล่งอ้างอิงที่ใช้ตอบ", expanded=False):
                for doc in docs:
                    meta = doc.metadata
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>Rank [{meta.get("rank", "-")}]</b> · '
                        f'Score: {meta.get("score", 0):.4f}<br>'
                        f'📜 {meta.get("law_name", "-")} มาตรา {meta.get("section_num", "-")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # -- บันทึก assistant message พร้อม sources --
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
    })