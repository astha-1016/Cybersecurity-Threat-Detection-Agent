# 🛡️ Cybersecurity Threat Detection Agent

> An AI-powered multi-node agent that analyzes security logs in real time using
> LangGraph, ChromaDB RAG, Groq LLaMA 3.3, and real threat intelligence APIs.

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────┐    ┌──────────┐    ┌──────────┐
│  Input  │───▶│  Memory  │───▶│  Intent  │
└─────────┘    └──────────┘    └──────────┘
                                     │
                    ┌────────────────┤
                    │                │
               "followup"       "new_query"
                    │                │
                    ▼                ▼
              ┌──────────┐    ┌──────────┐
              │ Follow-up│    │ Retrieve │ ◀── ChromaDB RAG
              │  (Memory)│    └──────────┘
              └────┬─────┘         │
                   │               ▼
                  END        ┌──────────┐
                             │ Decision │ ◀── LLM Classifier
                             └──────────┘
                                   │
                                   ▼
                             ┌──────────┐
                             │   Tool   │ ◀── AbuseIPDB + VirusTotal + NIST CVE
                             └──────────┘
                                   │
                                   ▼
                             ┌──────────┐
                             │ Response │ ◀── LLM Analysis
                             └──────────┘
                                   │
                                   ▼
                             ┌──────────┐
                             │   Eval   │ ◀── LLM Faithfulness Scoring
                             └──────────┘
                                   │
                    ┌──────────────┤
                    │              │
              faith < 0.7      faith ≥ 0.7
                    │              │
                    └──▶ retry ──┘ END
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-node LangGraph** | 9 nodes with conditional routing |
| **Intent detection** | Detects follow-up vs new query — answers from memory when appropriate |
| **ChromaDB RAG** | 10 cybersecurity knowledge base documents |
| **LLM Classification** | Groq LLaMA 3.3-70b classifies threat + attack type |
| **Real threat intelligence** | AbuseIPDB API + VirusTotal API + NIST NVD CVE lookup |
| **Self-reflection loop** | LLM scores its own faithfulness, retries if < 0.7 |
| **Persistent memory** | SQLite stores conversations and threat log across sessions |
| **Analytics dashboard** | Plotly charts: faithfulness over time, threat distribution, response time |
| **Threat log** | Persistent log of all confirmed threats |
| **Follow-up conversations** | Ask "how do I fix this?" or "explain simply" |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/yourname/cyber-threat-agent
cd cyber-threat-agent

# 2. Install dependencies
pip install -r requirements.txt

# Edit .env and add your GROQ_API_KEY

# 3. Add knowledge base documents
# Place your .txt files in data/docs/

# 4 . Run the test_cases
pytest tests/test_tools.py

# 5. Run
streamlit run app.py
```

---

## 🐳 Docker

```bash
docker build -t cyber-agent .
docker run -p 8501:8501 --env-file .env cyber-agent
```

---

## 🔑 API Keys

| Key | Required | Get it at |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | console.groq.com |
| `ABUSEIPDB_API_KEY` | Optional | abuseipdb.com |
| `VIRUSTOTAL_API_KEY` | Optional | virustotal.com |

---

## 📁 Project Structure

```
cyber-threat-agent/
├── app.py              # Streamlit UI (Chat + Analytics + Threat Log tabs)
├── graph.py            # LangGraph StateGraph
├── nodes.py            # All node functions
├── tools.py            # AbuseIPDB, VirusTotal, CVE lookup, anomaly detection
├── memory_store.py     # Persistent SQLite memory
├── config.py           # All configuration from .env
├── logger.py           # Structured logging
├── requirements.txt
├── Dockerfile
├── .env.example
└── data/
    ├── docs/           # Knowledge base .txt files (10 required)
    └── memory.db       # Auto-created SQLite database
```

---

## 🧪 Test Cases

| # | Input | Expected |
|---|---|---|
| 1 | `Multiple failed login attempts from 192.168.1.10` | 🔴 THREAT — Brute Force |
| 2 | `Unauthorized access in admin panel` | 🔴 THREAT — Unauthorized Access |
| 3 | `' OR 1=1 -- in login form` | 🔴 THREAT — SQL Injection |
| 4 | `Unknown software running automatically` | 🔴 THREAT — Malware |
| 5 | `Password reset email from unknown link` | 🔴 THREAT — Phishing |
| 6 | `Sudden spike in traffic causing slowdown` | 🟡 SUSPICIOUS — DDoS |
| 7 | `Login from new IP 172.16.0.2` | 🟡 SUSPICIOUS — Verify |
| 8 | `Multiple logins but eventually logged in` | 🟡 SUSPICIOUS |
| 9 | `User logged in successfully` | 🟢 SAFE |
| 10 | `Normal traffic during peak hours` | 🟢 SAFE |

---

## 📊 Evaluation

| Metric | Score |
|---|---|
| Faithfulness (avg) | ~0.82 |
| Test cases passed | 10/10 |
| Avg response time | ~4s |

---

## 🛠️ Tech Stack

- **LangGraph** — Multi-node agent orchestration
- **LangChain** — LLM integration and RAG
- **Groq** — LLaMA 3.3-70b inference
- **ChromaDB** — Vector store for RAG
- **HuggingFace** — Sentence embeddings (all-MiniLM-L6-v2)
- **AbuseIPDB** — IP reputation API
- **VirusTotal** — Second-opinion IP/domain analysis
- **NIST NVD** — CVE vulnerability database
- **Streamlit** — Web UI
- **SQLite** — Persistent memory storage
- **Plotly** — Analytics charts

---

## ⚠️ Limitations & Future Work

- Anomaly detection is pattern-based; a trained ML classifier (e.g., Random Forest on CICIDS2017) would be more accurate
- Knowledge base is static; production would integrate live threat feeds (MITRE ATT&CK, Shodan)
- Single-user; production would need authentication and multi-tenancy
- Would add email/Slack alerting for confirmed threats

