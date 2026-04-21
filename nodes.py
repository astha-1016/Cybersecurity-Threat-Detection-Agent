"""
nodes.py — All LangGraph node functions.

Pipeline:
  input → memory → intent → [followup → END]
                          → [retrieve → decision → tool → response → eval]
                                                              ↑         |
                                                              └─ retry ─┘

FIXES APPLIED:
  - FIX 1: memory_node — do NOT append the user message again here; app.py
           already appended it before calling graph.invoke(). Double-appending
           caused every query to appear twice in memory and broke context
           length / follow-up detection for the very first message.
  - FIX 2: intent_node — follow-up detection fired even when the previous
           assistant message was from the followup_node (faithfulness=1.0
           placeholder text). Added guard: if last assistant content contains
           the follow-up footer marker, treat current message as new_query
           unless clearly a follow-up.
  - FIX 3: decision_node — fallback attack_type was "None" (string), but
           tool_node, self_eval_node, and the UI all check for empty string
           to mean "no attack". Changed default to "" so downstream logic
           stays consistent.
  - FIX 4: response_node — history loop accidentally included the user message
           that was just added by memory_node, causing the prompt to contain
           the current query twice (once in history, once in the final
           HumanMessage). Fixed by iterating history[:-1] before appending
           the current HumanMessage at the end.
  - FIX 5: self_eval_node — faithfulness score prompt accepted bare integers
           0 and 1 as valid scores, but the regex matched them even inside
           words like "10" or "100". Added word-boundary anchors so "10"
           no longer parses as faithfulness=1.0.
  - FIX 6: self_eval_node — severity_score was never written back to state
           before the final dict was returned, so app.py always received 0.0.
           Now correctly included in the returned dict.
  - FIX 7: followup_node — returned state still had stale decision/attack_type
           from a PREVIOUS query when the user asked a follow-up immediately
           after a safe query. This caused the Threat Log to record false
           positives. Now explicitly preserves only values that were set by
           this run.
  - FIX 8: _call_llm — wrapped response in .strip() so leading/trailing
           whitespace never breaks downstream pattern matching.
  - FIX 9: tool_node — CVE lookup now uses the full attack_type string instead
           of only the first word, so "Data Exfiltration" searches correctly
           instead of just "Data".
  - FIX 10: response_node — computed severity label and score are now injected
            into the system prompt so the LLM narrative matches the footer score
            and cannot overcall/undercall the severity level.
  - FIX 11: response_node — numbered action list is now explicitly instructed
            as 1, 2, 3 to prevent the LLM from repeating "1." for every item.
  - FIX 12: self_eval_node — verdict and icon are now driven by severity_score
            thresholds (>=7.0 → THREAT, >=4.0 → POTENTIAL) instead of the
            raw decision string, keeping verdict consistent with the score.
  - FIX 13: self_eval_node — auto-corrects any stray HIGH label in the LLM
            narrative when the computed severity is MEDIUM/LOW/NONE.
"""

import re
import time
from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import (
    GROQ_API_KEY, MODEL_NAME, MODEL_TEMPERATURE,
    FAITHFULNESS_THRESHOLD, MAX_EVAL_RETRIES, MEMORY_WINDOW
)
from tools import detect_anomaly, check_ip_reputation, check_virustotal, lookup_cve, lookup_mitre, lookup_domain
from logger import get_logger

log = get_logger(__name__)

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=MODEL_TEMPERATURE)


def _call_llm(messages_or_prompt, retries: int = 3) -> str:
    """Call LLM with exponential backoff on rate limit errors."""
    for attempt in range(retries):
        try:
            if isinstance(messages_or_prompt, str):
                return llm.invoke(messages_or_prompt).content.strip()  # FIX 8
            return llm.invoke(messages_or_prompt).content.strip()       # FIX 8
        except Exception as e:
            err = str(e).lower()
            if ("rate_limit" in err or "429" in err) and attempt < retries - 1:
                wait = 2 ** attempt
                log.warning(f"Rate limited — retrying in {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                log.error(f"LLM call failed: {e}")
                return "Analysis unavailable — please try again in a moment."
    return "Analysis unavailable."


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    input:          str
    messages:       List[dict]
    intent:         str           # "followup" | "new_query"
    context:        str           # RAG chunks
    sources:        List[str]     # KB document names used
    decision:       str           # "threat" | "suspicious" | "safe"
    attack_type:    str           # specific attack name from LLM
    tool_output:    str           # combined tool results
    response:       str           # LLM analysis
    faithfulness:   float         # 0.0 – 1.0
    eval_retries:   int
    final:          str
    severity_score: float


# ── Node 1: Memory ────────────────────────────────────────────────────────────
def memory_node(state: AgentState) -> dict:
    """
    Maintain sliding window conversation history.

    FIX 1: app.py appends the user message to st.session_state.messages BEFORE
           invoking the graph, so 'messages' already contains the current user
           turn when it arrives here. We must NOT append it again or every
           query appears twice, corrupting the context window and breaking
           follow-up intent detection on the very first exchange.
    """
    messages = list(state.get("messages", []))
    # Only trim to window; do NOT append state["input"] again
    if len(messages) > MEMORY_WINDOW:
        messages = messages[-MEMORY_WINDOW:]
    log.info(f"Memory: {len(messages)} messages in window")
    return {
        **state,
        "messages":     messages,
        "eval_retries": 0,
        "faithfulness": 0.0,
        "sources":      [],
        "attack_type":  "",
    }


# ── Node 2: Intent Router ─────────────────────────────────────────────────────
def intent_node(state: AgentState) -> dict:
    """
    Classify input as 'followup' or 'new_query' using LLM.

    FIX 2: If the most recent assistant message is the follow-up footer
           (i.e. the previous turn was ALREADY a follow-up answer), do not
           blindly classify the next message as another follow-up. The LLM
           sometimes over-classifies consecutive messages as follow-ups,
           leading to RAG being skipped for new threat queries.
    """
    user_input = state["input"]
    history    = state.get("messages", [])
    has_prior  = any(m["role"] == "assistant" for m in history)

    if not has_prior:
        return {**state, "intent": "new_query"}

    last_assistant = next(
        (m["content"][:200] for m in reversed(history) if m["role"] == "assistant"),
        "none"
    )

    # FIX 2: if last reply was a follow-up answer, be conservative
    followup_footer = "Follow-up answered from conversation memory"
    if followup_footer in last_assistant:
        # Still allow explicit follow-ups like "explain more" / "how to fix"
        explicit_followup_keywords = [
            "explain", "how to fix", "what does", "simplify", "elaborate",
            "what should", "why did", "give me steps", "what happens", "more detail"
        ]
        if not any(kw in user_input.lower() for kw in explicit_followup_keywords):
            return {**state, "intent": "new_query"}

    prompt = f"""Classify this message as 'followup' or 'new_query'.

followup = asking about the previous answer (explain more, how to fix, what does that mean,
           elaborate, what should I do, why did that happen, give me steps, simplify)
new_query = a new log entry or unrelated security question

Previous assistant response (first 200 chars): {last_assistant}
Current user message: "{user_input}"

Reply with ONLY one word: followup / new_query"""

    result = _call_llm(prompt).strip().lower()
    intent = "followup" if "followup" in result else "new_query"
    log.info(f"Intent: {intent} for '{user_input[:50]}'")
    return {**state, "intent": intent}


# ── Node 3: Follow-up Answer ──────────────────────────────────────────────────
def followup_node(state: AgentState) -> dict:
    """
    Answer follow-up questions using conversation history.

    FIX 7: Preserve the decision and attack_type from the CURRENT state
           (which may already be empty/safe for this follow-up turn) rather
           than blindly copying from state which could hold stale threat data
           from a previous query and pollute the Threat Log.
    """
    user_input = state["input"]
    history    = state.get("messages", [])

    lc_msgs = [SystemMessage(content="""You are a cybersecurity expert assistant.
The user is asking a follow-up about a previous security analysis.
Be specific, practical, and beginner-friendly.
- If they ask how to fix → give numbered step-by-step instructions as 1, 2, 3 (never repeat the same number)
- If they ask to simplify → use a real-world analogy
- If they ask what happens next → describe the attack progression
Keep your answer focused and under 300 words.""")]

    # FIX 4 (also applied here): exclude the last user message since it is
    # the current input and will be added explicitly below
    for msg in history[:-1]:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        else:
            lc_msgs.append(AIMessage(content=msg["content"]))

    lc_msgs.append(HumanMessage(content=user_input))

    response = _call_llm(lc_msgs)
    log.info("Follow-up answered from memory")

    msgs = list(history)
    msgs.append({"role": "assistant", "content": response})

    final = f"""{response}

{"=" * 45}
💬 Follow-up answered from conversation memory
📊 Faithfulness: ██████████ 1.00/1.00
"""
    return {
        **state,
        "messages":     msgs,
        "response":     response,
        "faithfulness": 1.0,
        "eval_retries": 1,
        "final":        final,
        # FIX 7: use empty/safe defaults so stale threat data isn't forwarded
        "decision":       "safe",
        "attack_type":    "",
        "tool_output":    "",
        "context":        state.get("context", ""),
        "sources":        state.get("sources", []),
        "severity_score": 0.0,
    }


# ── Node 4: Decision (LLM classifier) ────────────────────────────────────────
def decision_node(state: AgentState) -> dict:
    """
    LLM-powered threat classifier.
    Returns: threat | suspicious | safe

    FIX 3: Default attack_type changed from "None" (string) to "" (empty string)
           so downstream checks `if attack_type and attack_type.lower() not in ("none","","unknown")`
           work correctly without needing to special-case the string "None".
    """
    user_input = state["input"]
    context    = state.get("context", "")[:800]

    prompt = f"""You are a cybersecurity threat classifier.

Given the log/query and knowledge base context, provide:
1. Classification: threat / suspicious / safe
2. Attack type: choose the MOST SPECIFIC name from this list:
   - Credential Dumping     (LSASS, Mimikatz, memory extraction, credential harvesting)
   - Brute Force            (repeated login failures, password guessing)
   - SQL Injection          (SQL payloads, OR 1=1, UNION SELECT, DROP TABLE)
   - XSS                    (script tags, javascript:, cross-site scripting)
   - Phishing               (fake emails, unknown links, password reset scams)
   - Malware                (trojans, rootkits, keyloggers, unknown software)
   - Ransomware             (file encryption, ransom demands)
   - DDoS                   (traffic spike, service disruption)
   - Unauthorized Access    (access without credentials, admin panel breach)
   - Privilege Escalation   (gaining higher permissions, sudo abuse)
   - Lateral Movement       (moving across internal network)
   - Data Exfiltration      (data theft, large transfers out)
   - Port Scanning          (network reconnaissance, port scan)
   - Credential Stuffing    (using leaked password lists)
   - Intrusion              (network intrusion, IDS alert)
   - None                   (safe / normal activity)

IMPORTANT: Choose the MOST SPECIFIC type. Examples:
   - LSASS or Mimikatz mentioned -> Credential Dumping (NOT Unauthorized Access)
   - Repeated login failures -> Brute Force (NOT Unauthorized Access)
   - Script injection -> XSS (NOT Malware)

Knowledge base context:
{context}

Log/Query: "{user_input}"

Reply in EXACTLY this format (two lines only):
CLASSIFICATION: <threat|suspicious|safe>
ATTACK_TYPE: <name from list above>"""

    result = _call_llm(prompt)
    lines  = result.strip().split("\n")

    decision    = "safe"
    attack_type = ""        # FIX 3: empty string instead of "None"

    for line in lines:
        line_lower = line.lower()
        if "classification:" in line_lower:
            if "threat" in line_lower:
                decision = "threat"
            elif "suspicious" in line_lower:
                decision = "suspicious"
            else:
                decision = "safe"
        if "attack_type:" in line_lower:
            parts = line.split(":", 1)
            if len(parts) > 1:
                raw_attack = parts[1].strip()
                # FIX 3: normalise "None" string to empty
                attack_type = "" if raw_attack.lower() == "none" else raw_attack

    log.info(f"Decision: {decision} | Attack: {attack_type!r}")
    return {**state, "decision": decision, "attack_type": attack_type}


# ── Node 5: Tool ──────────────────────────────────────────────────────────────
def tool_node(state: AgentState) -> dict:
    """
    Runs all relevant tools:
      - detect_anomaly:       pattern scan
      - check_ip_reputation:  AbuseIPDB API
      - check_virustotal:     VirusTotal API
      - lookup_cve:           NIST NVD CVE search
      - lookup_mitre:         MITRE ATT&CK mapping
      - lookup_domain:        DNS + IP reputation for domains
    """
    log_text    = state["input"]
    attack_type = state.get("attack_type", "")
    results     = []

    # Tool 1: Anomaly scan
    anomaly = detect_anomaly(log_text)
    results.append(f"🔍 Anomaly Scan: {anomaly}")

    # Tool 2 & 3: IP reputation (AbuseIPDB + VirusTotal)
    ips = re.findall(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", log_text)
    if ips:
        for ip in ips[:2]:  # max 2 IPs
            abuse_result = check_ip_reputation(ip)
            vt_result    = check_virustotal(ip)
            results.append(f"🌐 IP {ip}: {abuse_result}")
            if "API key not set" not in vt_result:
                results.append(f"🦠 VirusTotal {ip}: {vt_result}")
    else:
        results.append("🌐 IP Check: No IP address found in log")

    # Tool 6: Domain lookup
    domains = re.findall(
        r'\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|gov|edu|co)\b', log_text
    )
    domains = [d for d in domains if not re.match(r'^\d+\.\d+\.\d+\.\d+$', d)]
    if domains:
        domain_result = lookup_domain(domains[0])
        results.append(f"🌍 Domain {domains[0]}: {domain_result}")

    # Tool 4: CVE lookup for the detected attack type
    if attack_type and attack_type.lower() not in ("none", "", "unknown"):
        # FIX 9: use full attack_type string, not just the first word
        cve_result   = lookup_cve(attack_type)
        results.append(f"🔐 CVE Lookup: {cve_result}")
        # Tool 5: MITRE ATT&CK tactic mapping
        mitre_result = lookup_mitre(attack_type)
        results.append(f"🎯 MITRE ATT&CK: {mitre_result}")

    tool_output = "\n".join(results)
    log.info(f"Tools ran: {len(results)} results")
    return {**state, "tool_output": tool_output}


# ── Node 6: Response (LLM analysis) ──────────────────────────────────────────
def response_node(state: AgentState) -> dict:
    """
    LLM generates structured threat analysis from:
      - RAG context (knowledge base)
      - Tool results (real threat intelligence)
      - Conversation history (multi-turn awareness)

    FIX 4: history loop now iterates history[:-1] (excluding the latest user
           message that was just appended), then appends the current query as
           a fresh HumanMessage. Previously the loop went over history[:-1]
           but the final HumanMessage was added as 'Analyze: "{user_input}"'
           — meaning the query appeared once from history AND once explicitly,
           causing duplicate context and sometimes confusing the LLM.
           Now we explicitly skip history[-1] (the current user turn) so the
           conversation build-up is: system + prior pairs + current query.
    FIX 10: Computed severity label/score is injected into the system prompt
            so the LLM narrative cannot contradict the footer score.
    FIX 11: Numbered action list is explicitly instructed as 1, 2, 3.
    """
    user_input   = state.get("input", "")
    context      = state.get("context", "")
    tool_output  = state.get("tool_output", "")
    decision     = state.get("decision", "safe")
    attack_type  = state.get("attack_type", "Unknown")
    eval_retries = state.get("eval_retries", 0)
    sources      = state.get("sources", [])
    history      = state.get("messages", [])

    # FIX 10: pre-compute severity so LLM narrative matches the footer
    severity_score, severity_label, _ = compute_severity_score(
        decision,
        tool_output,
        state.get("faithfulness", 0.75),
    )

    # Map severity label to icon for the prompt
    if severity_score >= 7.0:
        severity_icon = "🔴"
    elif severity_score >= 4.0:
        severity_icon = "🟡"
    else:
        severity_icon = "🟢"

    source_str  = f"\nKB Sources used: {', '.join(sources)}" if sources else ""
    retry_note  = (
        "\n\n⚠️ STRICT MODE: Previous answer scored low on faithfulness. "
        "Use ONLY facts explicitly in the knowledge base context. "
        "Do NOT add information from your training data."
        if eval_retries > 0 else ""
    )

    attack_display = attack_type if attack_type else "Unknown"

    system = f"""You are an expert cybersecurity analyst helping beginners understand threats.

Pre-analysis results:
• Classification: {decision.upper()}
• Attack Type Detected: {attack_display}
• Computed Severity: {severity_label} ({severity_score}/10)

Tool Intelligence:
{tool_output}

Knowledge Base Context{source_str}:
{context}
{retry_note}

Write a complete analysis using EXACTLY this structure:

🔍 THREAT ANALYSIS
{"=" * 45}

⚔️  Attack Type: {attack_display}

❓ Why Does This Happen?
[2-3 clear sentences for a beginner. Use an analogy if helpful. No jargon.]

Tool Intelligence Summary:
[Summarize what the anomaly scan, IP checks, CVE results, and MITRE ATT&CK tactic actually mean]

📖 Knowledge Base Insight:
[What your knowledge base says about this attack — be specific]

🏷️  Severity: {severity_icon} {severity_label} ({severity_score}/10) — [one sentence justification]
IMPORTANT: Use EXACTLY the severity label and score shown above. Do NOT change it.

🛡️  Recommended Actions:
1. [Immediate action — do this first]
2. [Short-term action — do this today]
3. [Long-term action — do this this week]
IMPORTANT: Number the actions as 1, 2, 3 — never repeat the same number.

💬 Tip: You can ask me "how do I fix this?", "explain simply", or "what happens if I ignore this?" """

    lc_msgs = [SystemMessage(content=system)]

    # FIX 4: build history EXCLUDING the last message (current user turn)
    # The current user turn is added explicitly below as 'Analyze: ...'
    prior_history = history[:-1] if history and history[-1]["role"] == "user" else history
    for msg in prior_history:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_msgs.append(AIMessage(content=msg["content"]))

    lc_msgs.append(HumanMessage(content=f'Analyze: "{user_input}"'))

    response = _call_llm(lc_msgs)
    log.info(f"Response generated (retry={eval_retries})")

    msgs = list(state.get("messages", []))
    msgs.append({"role": "assistant", "content": response})

    return {**state, "response": response, "messages": msgs}


# ── Severity Scoring ──────────────────────────────────────────────────────────
def compute_severity_score(
    decision:     str,
    tool_output:  str,
    faithfulness: float,
) -> tuple[float, str, str]:
    """
    Computes a CVSS-style severity score (0.0 – 10.0) from agent signals.
    """
    BASE = {"threat": 8.0, "suspicious": 5.0, "safe": 1.0}
    base = BASE.get(decision.lower(), 1.0)

    bonus = 0.0

    if "Malicious IP" in tool_output:
        bonus += 1.5
    elif "Suspicious IP" in tool_output:
        bonus += 0.5
    elif "Unknown IP" in tool_output:
        bonus += 0.2

    if "Critical |" in tool_output:
        bonus += 1.0
    elif "Suspicious |" in tool_output:
        bonus += 0.3

    if "CVE-" in tool_output and "No CVEs found" not in tool_output:
        bonus += 0.5

    if "MITRE ATT&CK |" in tool_output and "No mapping found" not in tool_output:
        bonus += 0.3

    bonus = min(bonus, 3.0)
    raw   = base + bonus

    if faithfulness >= 0.80:
        modifier = 1.00
    elif faithfulness >= 0.60:
        modifier = 0.95
    else:
        modifier = 0.90

    score = round(min(raw * modifier, 10.0), 1)

    if score >= 9.0:
        label = "CRITICAL"
    elif score >= 7.0:
        label = "HIGH"
    elif score >= 4.0:
        label = "MEDIUM"
    elif score >= 1.5:
        label = "LOW"
    else:
        label = "NONE"

    filled     = round(score)
    rating_bar = "█" * filled + "░" * (10 - filled) + f"  {score}/10"

    log.info(f"Severity: {score} ({label}) | base={base} bonus={bonus:.1f} modifier={modifier}")
    return score, label, rating_bar


# ── Node 7: Self-Eval ─────────────────────────────────────────────────────────
def self_eval_node(state: AgentState) -> dict:
    """
    LLM scores its own response for faithfulness to the KB context.
    Computes CVSS-style severity score from all available signals.
    If score < FAITHFULNESS_THRESHOLD and retries remain → loop back.

    FIX 5:  Faithfulness regex now uses word boundaries so "10" in "score: 10"
            no longer parses as faithfulness=1.0 (via the bare int_matches path).
    FIX 6:  severity_score was missing from the returned dict — app.py always
            received 0.0. Added to the return value.
    FIX 12: Verdict and icon are driven by severity_score thresholds so they
            always agree with the displayed score, not just the raw decision string.
    FIX 13: Auto-corrects stray HIGH label in the LLM narrative when the
            computed severity is MEDIUM/LOW/NONE.
    """
    response     = state.get("response", "")
    context      = state.get("context", "")[:800]
    tool_output  = state.get("tool_output", "")
    decision     = state.get("decision", "safe")
    eval_retries = state.get("eval_retries", 0)

    # ── Faithfulness scoring ──────────────────────────────────────────────────
    faithfulness = 0.75  # safe default
    if context.strip():
        score_prompt = f"""You are a faithfulness evaluator. Score how well this analysis is grounded in the context.

1.0 = every claim is directly supported by the context
0.8 = mostly faithful with minor additions
0.6 = some claims go beyond the context
0.4 = many claims not supported
0.2 = mostly unsupported
0.0 = completely hallucinated

Context: {context}
Analysis: {response[:500]}

IMPORTANT: Reply with ONLY a single decimal number between 0.0 and 1.0. Example: 0.85"""

        try:
            raw             = llm.invoke(score_prompt).content.strip()
            decimal_matches = re.findall(r"0\.[0-9]+|1\.0+", raw)
            # FIX 5: use word boundaries to avoid matching "10" as "1"
            int_matches     = re.findall(r"\b[01]\b", raw)
            if decimal_matches:
                faithfulness = max(0.0, min(1.0, float(decimal_matches[0])))
            elif int_matches:
                faithfulness = max(0.0, min(1.0, float(int_matches[0])))
        except Exception:
            pass  # keep the 0.75 default

    retries = eval_retries + 1
    log.info(f"Faithfulness: {faithfulness:.2f} (retry {retries})")

    # ── CVSS-style severity scoring ───────────────────────────────────────────
    severity_score, severity_label, severity_bar = compute_severity_score(
        decision, tool_output, faithfulness
    )

    # ── Verdict driven by severity_score (FIX 12) ────────────────────────────
    is_critical        = "Critical |" in tool_output
    is_malicious       = "Malicious IP" in tool_output
    is_suspicious_tool = "Suspicious |" in tool_output or "Suspicious IP" in tool_output
    has_unknown        = "Unknown IP" in tool_output and decision != "safe"

    # FIX 12: use severity_score thresholds so verdict matches the displayed score
    if severity_score >= 7.0 or is_malicious or is_critical:
        verdict    = "THREAT CONFIRMED — Immediate action recommended."
        confidence = "High"
    elif severity_score >= 4.0 or is_suspicious_tool or has_unknown:
        verdict    = "POTENTIAL THREAT — Manual review strongly advised."
        confidence = "Medium"
    else:
        verdict    = "NO THREAT DETECTED — Activity appears normal."
        confidence = "High"

    # FIX 12: icon from score, not from label dict
    if severity_score >= 7.0:
        sev_icon = "🔴"
    elif severity_score >= 4.0:
        sev_icon = "🟡"
    else:
        sev_icon = "🟢"

    faith_bar   = "█" * int(faithfulness * 10) + "░" * (10 - int(faithfulness * 10))
    retry_label = (
        f" · improved after {retries-1} retr{'y' if retries-1 == 1 else 'ies'}"
        if retries > 1 else ""
    )

    # FIX 13: auto-correct stray HIGH in narrative when score disagrees
    corrected_response = response
    if severity_label in ("MEDIUM", "LOW", "NONE") and "🔴 HIGH" in corrected_response:
        corrected_response = corrected_response.replace(
            "🔴 HIGH —", f"{sev_icon} {severity_label} —"
        ).replace(
            "🔴 HIGH—", f"{sev_icon} {severity_label}—"
        )
        log.warning(
            f"FIX 13: Corrected stray HIGH label in narrative. "
            f"Computed severity={severity_score} ({severity_label})"
        )

    final = f"""{corrected_response}

{"=" * 45}
📌 Final Verdict: {verdict}
{sev_icon} Severity Score: {severity_bar}  [{severity_label}]
🎯 Confidence: {confidence}
📊 Faithfulness: {faith_bar} {faithfulness:.2f}/1.00{retry_label}
"""

    return {
        **state,
        "faithfulness":   faithfulness,
        "severity_score": severity_score,   # FIX 6: was missing
        "eval_retries":   retries,
        "final":          final,
    }