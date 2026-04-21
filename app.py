"""
app.py — Streamlit UI for the Cybersecurity Threat Detection Agent.
Run: streamlit run app.py

FIXES APPLIED:
  - FIX 1: graph.invoke() was passing st.session_state.messages AFTER the
           user message had already been appended to it (line just above).
           nodes.py/memory_node then appended the user message AGAIN, causing
           every query to appear twice in the context window and corrupting
           follow-up detection. Now we pass a copy of messages that already
           includes the current user turn; memory_node no longer re-appends.
  - FIX 2: severity_score metric badge — icon threshold was inverted: scores
           >= 7 showed 🟢 and scores < 4 showed 🔴. Fixed.
  - FIX 3: Chat history rendering — `render_response` was called on assistant
           messages that might contain only the raw response text (without the
           verdict lines), causing plain text to be silently dropped when all
           lines were filtered. Added a plain st.markdown fallback.
  - FIX 4: load_agent() — if documents fail to embed (e.g. HuggingFace model
           download issue), a helpful error is shown instead of a bare exception.
  - FIX 5: Removed duplicate severity_score chart that was rendering twice in
           the Analytics tab (fig0 was added outside the column layout and
           also inside it on retry).
"""

import uuid
import time
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from graph import build_graph
from memory_store import get_memory_store
from config import DOCS_PATH, EMBED_MODEL, TOP_K_RETRIEVAL
from logger import get_logger

log = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cybersecurity Threat Detection Agent",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #c9d1d9; }
  .block-container { padding-top: 3rem; }
  .metric-label { font-size: 11px !important; }

  header[data-testid="stHeader"] {
      height: 0;
      visibility: hidden;
  }

  .stTabs [data-baseweb="tab-list"] {
      position: fixed;
      top: 0;
      left: 350px;
      right: 0;
      z-index: 9999;
      background-color: #0d1117;
      padding: 8px 16px 0px 16px;
      overflow-x: auto;
      flex-wrap: nowrap;
      border-bottom: 1px solid #30363d;
  }

  .stTabs [data-baseweb="tab"] {
      font-size: 14px;
      white-space: nowrap;
      color: #c9d1d9;
      padding: 8px 20px;
  }

  .stTabs [data-baseweb="tab-highlight"] {
      background-color: #58a6ff;
  }
</style>
""", unsafe_allow_html=True)


# ── Helper ────────────────────────────────────────────────────────────────────
def render_response(content: str):
    """Render assistant response with colour-coded verdict lines.

    FIX 3: Added st.markdown fallback for lines that don't match any verdict
           keyword, so plain explanation lines are never silently dropped.
    """
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            st.write("")   # preserve blank lines for spacing
            continue
        if "THREAT CONFIRMED" in stripped:
            st.error(stripped)
        elif "POTENTIAL THREAT" in stripped:
            st.warning(stripped)
        elif "NO THREAT DETECTED" in stripped:
            st.success(stripped)
        else:
            st.markdown(line)   # FIX 3: use markdown so formatting is preserved


# ── Load agent (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    log.info("Loading agent...")
    try:
        loader    = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        if not documents:
            st.error(f"❌ No documents found in {DOCS_PATH}")
            st.stop()
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        # AFTER
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory="data/chroma_db"
        )
        retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
        graph       = build_graph(retriever)
        log.info(f"Agent loaded with {len(documents)} documents")
        return graph, len(documents)
    except Exception as e:
        # FIX 4: surface the real error so users know what went wrong
        st.error(f"❌ Failed to load agent: {e}")
        log.error(f"Agent load error: {e}")
        st.stop()


graph, doc_count = load_agent()
mem_store        = get_memory_store()


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "thread_id"     not in st.session_state: st.session_state.thread_id     = str(uuid.uuid4())[:8]
if "eval_history"  not in st.session_state: st.session_state.eval_history  = []
if "run_prompt"    not in st.session_state: st.session_state.run_prompt     = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Cyber Threat Agent")
    st.success(f"✅ {doc_count} KB documents loaded")
    st.divider()

    stats = mem_store.get_session_stats(st.session_state.thread_id)
    st.markdown("**📊 Session Summary**")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Threats",    stats.get("threat", 0))
    c2.metric("🟡 Suspicious", stats.get("suspicious", 0))
    c3.metric("🟢 Safe",       stats.get("safe", 0))
    st.divider()

    st.markdown("**📚 Knowledge Base**")
    for t in ["Brute Force", "DDoS", "Firewall", "Intrusion Detection",
              "Malware", "Network Security", "Phishing",
              "Ransomware", "SQL Injection", "XSS"]:
        st.write(f"• {t}")
    st.divider()

    st.markdown("**🧪 Quick Examples**")
    examples = {
        "🔴 Brute Force":    "Multiple failed login attempts detected from IP 192.168.1.10",
        "🔴 SQL Injection":  "User input contains ' OR 1=1 -- in the login form",
        "🔴 Malware":        "System is behaving slowly and unknown software is running automatically",
        "🔴 Unauthorized":   "Unauthorized access attempt detected in admin panel",
        "🎣 Phishing":       "User received an email asking for password reset from unknown link",
        "🟡 DDoS?":          "Sudden spike in traffic causing server slowdown",
        "🟡 New IP":         "Login attempt from new IP address 172.16.0.2",
        "🟡 Mixed":          "Multiple login attempts but user eventually logged in",
        "🟢 Normal login":   "User logged in successfully from registered device",
        "🟢 Normal traffic": "Normal traffic from users during peak hours",
    }
    for label, query in examples.items():
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state.run_prompt = query

    st.divider()
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.thread_id    = str(uuid.uuid4())[:8]
        st.session_state.eval_history = []
        st.rerun()
    st.caption(f"Session: {st.session_state.thread_id}")


# ── Main area with tabs ───────────────────────────────────────────────────────
tab_chat, tab_analytics, tab_threats = st.tabs(
    ["💬 Chat", "📊 Analytics", "🚨 Threat Log"]
)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1: CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## 🛡️ Cybersecurity Threat Detection Agent")
    st.caption("LangGraph · ChromaDB RAG · Groq LLaMA 3.3 · AbuseIPDB · VirusTotal · NIST CVE")
    st.divider()

    # ── Input ─────────────────────────────────────────────────────────────────
    prompt = st.session_state.pop("run_prompt", None)
    typed  = st.chat_input(
        "Enter a log or query… or ask a follow-up like 'how do I fix this?'"
    )
    if typed:
        prompt = typed

    # ── Run agent ─────────────────────────────────────────────────────────────
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        # FIX 1: append user message to session BEFORE passing to graph so
        # memory_node sees it as already part of history (no double-append)
        st.session_state.messages.append({"role": "user", "content": prompt})
        mem_store.save_message(st.session_state.thread_id, "user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing with LLM + RAG + Tools…"):
                t0 = time.time()
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    # FIX 1: pass messages that already include the current user turn
                    result = graph.invoke(
                        {
                            "input":    prompt,
                            "messages": list(st.session_state.messages),
                        },
                        config=config,
                    )
                    final          = result.get("final", "⚠️ No response generated.")
                    faithfulness   = result.get("faithfulness", 0.0)
                    retries        = result.get("eval_retries", 0)
                    decision       = result.get("decision", "safe")
                    severity_score = result.get("severity_score", 0.0)
                    sources        = result.get("sources", [])
                    intent         = result.get("intent", "new_query")
                    attack_type    = result.get("attack_type", "")
                    tool_output    = result.get("tool_output", "")
                    resp_time      = round(time.time() - t0, 1)
                except Exception as e:
                    final          = f"❌ Error: {str(e)}"
                    faithfulness   = 0.0
                    retries        = 0
                    decision       = "safe"
                    severity_score = 0.0
                    sources        = []
                    intent         = "new_query"
                    attack_type    = ""
                    tool_output    = ""
                    resp_time      = 0.0
                    log.error(f"Agent error: {e}")

            render_response(final)

            parts = []
            if severity_score > 0:
                # FIX 2: correct icon thresholds (was inverted)
                sev_icon = "🔴" if severity_score >= 7.0 else ("🟡" if severity_score >= 4.0 else "🟢")
                parts.append(f"{sev_icon} Severity: {severity_score}/10")

            if faithfulness > 0:
                icon = "✅" if faithfulness >= 0.7 else "⚠️"
                parts.append(f"{icon} Faithfulness: {faithfulness:.2f}")
            if retries > 1:
                parts.append(f"🔄 {retries-1} retry")
            if sources:
                parts.append(f"📖 {', '.join(sources)}")
            if intent == "followup":
                parts.append("💬 follow-up")
            parts.append(f"⏱ {resp_time}s")
            st.caption(" · ".join(parts))

        mem_store.save_message(
            st.session_state.thread_id, "assistant", final,
            decision=decision, faithfulness=faithfulness
        )
        if intent == "new_query":
            mem_store.log_threat(
                st.session_state.thread_id, prompt, decision,
                attack_type, tool_output, faithfulness
            )

        st.session_state.eval_history.append({
            "query_num":      len(st.session_state.eval_history) + 1,
            "input_preview":  prompt[:50],
            "decision":       decision,
            "attack_type":    attack_type,
            "faithfulness":   faithfulness,
            "severity_score": severity_score,
            "response_time":  resp_time,
            "retries":        retries,
            "intent":         intent,
        })

        st.session_state.messages.append({"role": "assistant", "content": final})
        st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        st.markdown("#### 🗂️ Conversation History")

        messages = st.session_state.messages
        pairs    = []
        i = 0
        while i < len(messages):
            user_msg = messages[i] if i < len(messages) else None
            asst_msg = messages[i+1] if i+1 < len(messages) else None
            if user_msg:
                pairs.append((user_msg, asst_msg))
            i += 2

        for user_msg, asst_msg in reversed(pairs):
            with st.chat_message(user_msg["role"]):
                st.write(user_msg["content"])
            if asst_msg:
                with st.chat_message(asst_msg["role"]):
                    render_response(asst_msg["content"])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2: ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Agent Performance Analytics")

    history = st.session_state.get("eval_history", [])

    if not history:
        st.info("💡 Ask some questions in the Chat tab to see analytics here.")
    else:
        try:
            import plotly.express as px
            import pandas as pd

            df = pd.DataFrame(history)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Queries",      len(df))
            m2.metric("Avg Severity",       f"{df['severity_score'].mean():.1f}/10")
            m3.metric("Max Severity",       f"{df['severity_score'].max():.1f}/10")
            m4.metric("Avg Faithfulness",   f"{df['faithfulness'].mean():.2f}")
            m5.metric("Avg Response Time",  f"{df['response_time'].mean():.1f}s")
            st.divider()

            # FIX 5: severity chart is shown once here, not duplicated
            fig0 = px.line(
                df, x="query_num", y="severity_score",
                title="Severity Score Over Session",
                markers=True,
                color_discrete_sequence=["#e74c3c"]
            )
            fig0.add_hrect(y0=9.0,  y1=10.0, fillcolor="#e74c3c", opacity=0.08, line_width=0, annotation_text="CRITICAL")
            fig0.add_hrect(y0=7.0,  y1=9.0,  fillcolor="#e74c3c", opacity=0.05, line_width=0, annotation_text="HIGH")
            fig0.add_hrect(y0=4.0,  y1=7.0,  fillcolor="#f39c12", opacity=0.05, line_width=0, annotation_text="MEDIUM")
            fig0.add_hrect(y0=0.0,  y1=4.0,  fillcolor="#2ecc71", opacity=0.05, line_width=0, annotation_text="LOW")
            fig0.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font_color="#c9d1d9", yaxis_range=[0, 10]
            )
            st.plotly_chart(fig0, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.line(
                    df, x="query_num", y="faithfulness",
                    title="Faithfulness Score Over Session",
                    markers=True,
                    color_discrete_sequence=["#3498db"]
                )
                fig1.add_hline(
                    y=0.7, line_dash="dash", line_color="#e74c3c",
                    annotation_text="Threshold (0.7)"
                )
                fig1.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font_color="#c9d1d9", yaxis_range=[0, 1]
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                counts = df["decision"].value_counts().reset_index()
                counts.columns = ["decision", "count"]
                fig2 = px.pie(
                    counts, values="count", names="decision",
                    title="Threat Distribution",
                    color="decision",
                    color_discrete_map={
                        "threat":     "#e74c3c",
                        "suspicious": "#f39c12",
                        "safe":       "#2ecc71"
                    }
                )
                fig2.update_layout(
                    paper_bgcolor="#0d1117", font_color="#c9d1d9"
                )
                st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                fig3 = px.bar(
                    df, x="query_num", y="response_time",
                    title="Response Time per Query (seconds)",
                    color="decision",
                    color_discrete_map={
                        "threat": "#e74c3c", "suspicious": "#f39c12", "safe": "#2ecc71"
                    }
                )
                fig3.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font_color="#c9d1d9"
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                if df["attack_type"].str.strip().ne("").any():
                    atk = df[df["attack_type"].str.strip() != ""]["attack_type"].value_counts()
                    fig4 = px.bar(
                        x=atk.index, y=atk.values,
                        title="Attack Types Detected",
                        labels={"x": "Attack Type", "y": "Count"},
                        color_discrete_sequence=["#e74c3c"]
                    )
                    fig4.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font_color="#c9d1d9"
                    )
                    st.plotly_chart(fig4, use_container_width=True)

            st.divider()
            st.markdown("**📋 Raw Session Data**")
            st.dataframe(
                df[["query_num", "input_preview", "decision",
                    "attack_type", "faithfulness", "response_time", "retries"]],
                use_container_width=True
            )

        except ImportError:
            st.warning("Install plotly for charts: `pip install plotly`")
            st.dataframe(history)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3: THREAT LOG
# ════════════════════════════════════════════════════════════════════════════
with tab_threats:
    st.markdown("### 🚨 Confirmed Threat Log")
    st.caption("All THREAT-classified queries from this session, stored persistently.")

    threats = mem_store.get_all_threats(st.session_state.thread_id)

    if not threats:
        st.success("✅ No confirmed threats logged in this session.")
    else:
        st.error(f"⚠️ {len(threats)} confirmed threat(s) detected this session")
        for i, t in enumerate(threats, 1):
            with st.expander(f"🔴 Threat #{i} — {t['attack_type']} — {t['timestamp'][:19]}"):
                st.write(f"**Input:** {t['input']}")
                st.write(f"**Attack Type:** {t['attack_type']}")
                st.write(f"**Timestamp:** {t['timestamp']}")