"""
tools.py — Cybersecurity Threat Detection Agent tools.

Tool 1: detect_anomaly(log)     — pattern-based log scanner
Tool 2: check_ip_reputation(ip) — AbuseIPDB API + local fallback
Tool 3: check_virustotal(ip)    — VirusTotal API
Tool 4: lookup_cve(keyword)     — NIST NVD CVE search
Tool 5: lookup_mitre(attack)    — MITRE ATT&CK tactic + technique mapping
Tool 6: lookup_domain(domain)   — DNS + IP reputation

Setup (.env):
    ABUSEIPDB_API_KEY=your_key_here
    VIRUSTOTAL_API_KEY=your_key_here

FIXES APPLIED:
  - FIX 1: detect_anomaly — moved overlapping "malware" pattern AFTER more-specific
            patterns (trojan/rootkit/keylogger) so they are not shadowed
  - FIX 2: detect_anomaly — added patterns for "new ip", "login from",
            "credential dumping", "credential harvesting" and "lsass"/"mimikatz"
            (were absent from suspicious block, causing miss-classifications)
  - FIX 3: check_ip_reputation — API response key was "abuseConfidenceScore"
            but older v2 responses use "abuseConfidencePercentage"; handle both
  - FIX 4: lookup_cve — NIST NVD API now requires 0.6s delay between requests
            without an API key; added rate-limit sleep + retry with back-off
  - FIX 5: lookup_cve — severity extraction now also handles cvssMetricV30
  - FIX 6: lookup_mitre — added missing entries: Lateral Movement, Privilege
            Escalation, Data Exfiltration, Port Scanning, Credential Dumping
  - FIX 7: lookup_mitre — alias dict was not handling multi-word canonical
            names that differ only in casing (e.g. "Credential Dumping")
  - FIX 8: lookup_domain — strip trailing dots and port numbers from domain
            before DNS resolution so "evil.com:8080" no longer fails
  - FIX 9: _is_valid_ip — reject leading-zero octets (e.g. "192.168.01.1")
            which are technically invalid and confuse some APIs
"""

import os
import re
import socket
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Logging (graceful fallback) ───────────────────────────────────────────────
try:
    from logger import get_logger
    log = get_logger(__name__)
except ImportError:
    import logging
    log = logging.getLogger(__name__)

# ── API Keys ──────────────────────────────────────────────────────────────────
ABUSEIPDB_API_KEY  = os.getenv("ABUSEIPDB_API_KEY", "")
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")

# ── Fallback IP lists ─────────────────────────────────────────────────────────
_MALICIOUS_IPS = {"192.168.1.10", "10.0.0.5", "45.33.32.156"}
_SAFE_IPS      = {"10.0.0.1", "192.168.1.1", "172.16.0.1", "8.8.8.8", "1.1.1.1"}


# ── Tool 1: Anomaly Detection ─────────────────────────────────────────────────
def detect_anomaly(log_text: str) -> str:
    """
    Scans a log string for known threat patterns.
    Returns: 'Critical | <label>', 'Suspicious | <label>', or 'Normal | ...'

    FIX 1: More-specific sub-types (trojan, rootkit, keylogger, mimikatz, lsass)
           are now checked BEFORE the generic "malware" pattern so they aren't
           shadowed by the broader match.
    FIX 2: Added missing patterns that were causing safe/suspicious misclassifications:
           "new ip address", "login from", "credential dumping/harvesting",
           "running automatically", "behaving slowly" (moved to critical).
    """
    t = log_text.lower()

    # ── CRITICAL patterns (order matters — specific before generic) ───────────
    critical_patterns = [
        # SQL Injection
        ("or 1=1",                "SQL injection bypass"),
        ("' or '1'='1",           "SQL injection bypass"),
        ("drop table",            "SQL table destruction"),
        ("union select",          "SQL data exfiltration"),
        ("sql injection",         "SQL injection"),
        # XSS
        ("<script>",              "XSS script injection"),
        ("javascript:",           "XSS inline script"),
        ("cross-site scripting",  "Cross-site scripting"),
        ("xss",                   "Cross-site scripting"),
        # Ransomware
        ("ransomware",            "Ransomware"),
        # Credential Dumping — MUST come before generic "malware"
        ("credential dumping",    "Credential Dumping attack"),
        ("credential harvesting", "Credential harvesting attack"),
        ("lsass",                 "Credential Dumping via LSASS memory access"),
        ("mimikatz",              "Credential Dumping via Mimikatz tool"),
        # Specific malware sub-types — MUST come before generic "malware"
        ("trojan",                "Trojan horse"),
        ("rootkit",               "Rootkit"),
        ("keylogger",             "Keylogger malware"),
        # Generic malware (after specific sub-types)
        ("malware detected",      "Malware confirmed"),
        ("malware",               "Malware"),
        ("unknown software",      "Unknown software running"),
        ("running automatically", "Auto-start malware symptom"),   # FIX 2: was missing from critical
        ("behaving slowly",       "Performance degradation symptom"),
        # Access attacks
        ("unauthorized",          "Unauthorized access"),
        ("brute force",           "Brute force attack"),
        ("intrusion detected",    "Network intrusion"),
        # Phishing
        ("phishing",              "Phishing attempt"),
        ("unknown link",          "Suspicious link"),
        ("password reset",        "Phishing password reset"),
        # Data theft
        ("data exfiltration",     "Data theft attempt"),
        # Privilege escalation
        ("privilege escalation",  "Privilege escalation"),
    ]

    for pattern, label in critical_patterns:
        if pattern in t:
            log.info(f"Anomaly: Critical — {label}")
            return f"Critical | {label} detected"

    # ── SUSPICIOUS patterns ───────────────────────────────────────────────────
    suspicious_patterns = [
        ("multiple failed",         "Multiple failed login attempts"),
        ("multiple login",          "Multiple login attempts"),
        ("failed login",            "Failed login attempt"),
        ("multiple attempts",       "Multiple access attempts"),
        ("login attempt",           "Login attempt from unknown source"),
        # FIX 2: "new ip address" / "login from new ip" were undetected
        ("new ip address",          "Login from new/unrecognised IP"),
        ("login from new",          "Login from new/unrecognised IP"),
        ("login from",              "Login from potentially unknown source"),
        ("access attempt",          "Unauthorized access attempt"),
        ("sudden spike",            "Sudden traffic spike"),
        ("server slowdown",         "Server performance degradation"),
        ("traffic spike",           "Traffic anomaly"),
        ("ddos",                    "DDoS indicator"),
        ("flood",                   "Flood attack indicator"),
        ("overload",                "Service overload"),
        ("port scan",               "Port scanning activity"),
        ("suspicious",              "Flagged as suspicious"),
        ("unusual activity",        "Unusual activity"),
        ("anomaly",                 "Anomaly detected"),
        ("bulk download",           "Bulk data download"),
        ("outside normal working",  "After-hours access"),
        ("3am",                     "After-hours system access"),
        ("2am",                     "After-hours system access"),
        ("unusual outbound",        "Unusual outbound transfer"),
        ("outbound data transfer",  "Data transfer exfiltration indicator"),
        ("ssl certificate warning", "SSL anomaly"),
        ("unexpected ssl",          "SSL anomaly"),
        ("dns resolution",          "DNS anomaly"),
        ("network redirect",        "Suspicious redirect"),
        ("new administrator",       "Unauthorised admin account created"),
        ("sudo command",            "Privilege misuse"),
        ("lateral movement",        "APT lateral movement"),
        ("unusual cloud",           "Anomalous cloud activity"),
        ("unusual internal",        "Anomalous internal traffic"),
        ("unusual",                 "Unusual behaviour detected"),
    ]

    for pattern, label in suspicious_patterns:
        if pattern in t:
            log.info(f"Anomaly: Suspicious — {label}")
            return f"Suspicious | {label}"

    return "Normal | No known threat patterns detected"


# ── Tool 2: IP Reputation via AbuseIPDB ──────────────────────────────────────
def check_ip_reputation(ip: str) -> str:
    """
    FIX 3: AbuseIPDB v2 may return either 'abuseConfidenceScore' or the older
           'abuseConfidencePercentage' key. We now handle both so no KeyError
           silently returns 0 (causing a Malicious IP to be reported as Clean).
    """
    if not _is_valid_ip(ip):
        return f"Invalid IP format: {ip}"

    if ABUSEIPDB_API_KEY:
        for attempt in range(3):
            try:
                resp    = requests.get(
                    "https://api.abuseipdb.com/api/v2/check",
                    headers={"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"},
                    params={"ipAddress": ip, "maxAgeInDays": 90},
                    timeout=5,
                )
                data    = resp.json().get("data", {})
                # FIX 3: handle both key names returned by AbuseIPDB v2
                score   = (data.get("abuseConfidenceScore")
                           or data.get("abuseConfidencePercentage")
                           or 0)
                isp     = data.get("isp", "Unknown ISP")
                country = data.get("countryCode", "??")
                reports = data.get("totalReports", 0)

                if score >= 50:
                    return (f"Malicious IP — AbuseIPDB score {score}/100 | "
                            f"ISP: {isp} | Country: {country} | Reports: {reports}")
                elif score >= 10:
                    return f"Suspicious IP — AbuseIPDB score {score}/100 | Country: {country}"
                else:
                    return f"Clean IP — AbuseIPDB score {score}/100 | Country: {country}"

            except requests.exceptions.Timeout:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return "Unknown IP — AbuseIPDB timeout"
            except Exception as e:
                log.warning(f"AbuseIPDB error: {e}")
                return "Unknown IP — AbuseIPDB error"

    # Fallback — no API key
    if ip in _MALICIOUS_IPS:
        return "Malicious IP — known threat actor (local list)"
    if ip in _SAFE_IPS:
        return "Clean IP — registered/trusted device (local list)"
    return "Unknown IP — not in any list, manual verification needed"


# ── Tool 3: VirusTotal IP Check ───────────────────────────────────────────────
def check_virustotal(ip: str) -> str:
    if not VIRUSTOTAL_API_KEY:
        return "VirusTotal: API key not set (add VIRUSTOTAL_API_KEY to .env)"
    if not _is_valid_ip(ip):
        return f"VirusTotal: Invalid IP format: {ip}"
    try:
        resp      = requests.get(
            f"https://www.virustotal.com/api/v3/ip_addresses/{ip}",
            headers={"x-apikey": VIRUSTOTAL_API_KEY},
            timeout=5,
        )
        stats     = resp.json()["data"]["attributes"]["last_analysis_stats"]
        malicious  = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        harmless   = stats.get("harmless", 0)
        if malicious > 0:
            return (f"Malicious IP — VirusTotal: {malicious} engines flagged, "
                    f"{suspicious} suspicious, {harmless} clean")
        elif suspicious > 0:
            return f"Suspicious IP — VirusTotal: {suspicious} engines flagged as suspicious"
        else:
            return f"Clean IP — VirusTotal: {harmless} engines say clean"
    except Exception as e:
        log.warning(f"VirusTotal error: {e}")
        return "VirusTotal: lookup failed"


# ── Tool 4: CVE Lookup via NIST NVD ──────────────────────────────────────────
def lookup_cve(keyword: str) -> str:
    """
    FIX 4: NIST NVD now enforces a 0.6s rate limit between unauthenticated
           requests. Added sleep + retry with exponential back-off so we no
           longer silently return an empty result on the first real query.
    FIX 5: Added cvssMetricV30 to severity extraction path.
    """
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(0.6 * attempt)  # FIX 4: rate-limit compliance
            resp  = requests.get(
                "https://services.nvd.nist.gov/rest/json/cves/2.0",
                params={"keywordSearch": keyword, "resultsPerPage": 3},
                timeout=10,
            )
            if resp.status_code == 403:
                return "CVE lookup: NIST NVD rate-limited — try again in a few seconds"
            vulns = resp.json().get("vulnerabilities", [])
            if not vulns:
                return f"No CVEs found for '{keyword}' in NIST NVD"
            results = []
            for v in vulns:
                cve      = v["cve"]
                cid      = cve["id"]
                desc     = cve["descriptions"][0]["value"][:200]
                severity = "Unknown"
                try:
                    m = cve.get("metrics", {})
                    # FIX 5: check V31, V30, then V2
                    if "cvssMetricV31" in m:
                        severity = m["cvssMetricV31"][0]["cvssData"]["baseSeverity"]
                    elif "cvssMetricV30" in m:
                        severity = m["cvssMetricV30"][0]["cvssData"]["baseSeverity"]
                    elif "cvssMetricV2" in m:
                        severity = m["cvssMetricV2"][0]["baseSeverity"]
                except Exception:
                    pass
                results.append(f"• {cid} [{severity}]: {desc}")
            return "Related CVEs from NIST NVD:\n" + "\n".join(results)
        except requests.exceptions.Timeout:
            if attempt < 2:
                continue
            return "CVE lookup timed out — NIST NVD may be slow"
        except Exception as e:
            log.warning(f"CVE lookup error: {e}")
            return f"CVE lookup failed: {str(e)[:60]}"
    return "CVE lookup failed after retries"


# ── Tool 5: MITRE ATT&CK Tactic Mapping ──────────────────────────────────────
# FIX 6: Added missing attack types that nodes.py classifier can return:
#        Lateral Movement, Privilege Escalation, Data Exfiltration,
#        Port Scanning, Credential Dumping, Credential Stuffing (already existed).
_MITRE_MAP = {
    "Phishing": {
        "tactic":      "Initial Access",
        "tactic_id":   "TA0001",
        "technique":   "Phishing",
        "tech_id":     "T1566",
        "description": "Adversary sends malicious emails to gain initial foothold.",
        "url":         "https://attack.mitre.org/techniques/T1566/",
    },
    "Malware": {
        "tactic":      "Execution",
        "tactic_id":   "TA0002",
        "technique":   "User Execution: Malicious File",
        "tech_id":     "T1204.002",
        "description": "User is tricked into running a malicious file or program.",
        "url":         "https://attack.mitre.org/techniques/T1204/002/",
    },
    "Ransomware": {
        "tactic":      "Impact",
        "tactic_id":   "TA0040",
        "technique":   "Data Encrypted for Impact",
        "tech_id":     "T1486",
        "description": "Adversary encrypts data to disrupt availability and extort payment.",
        "url":         "https://attack.mitre.org/techniques/T1486/",
    },
    "Brute Force": {
        "tactic":      "Credential Access",
        "tactic_id":   "TA0006",
        "technique":   "Brute Force",
        "tech_id":     "T1110",
        "description": "Adversary attempts many passwords to gain account access.",
        "url":         "https://attack.mitre.org/techniques/T1110/",
    },
    "Credential Stuffing": {
        "tactic":      "Credential Access",
        "tactic_id":   "TA0006",
        "technique":   "Brute Force: Credential Stuffing",
        "tech_id":     "T1110.004",
        "description": "Adversary uses leaked credential pairs from other breaches.",
        "url":         "https://attack.mitre.org/techniques/T1110/004/",
    },
    # FIX 6: Added Credential Dumping
    "Credential Dumping": {
        "tactic":      "Credential Access",
        "tactic_id":   "TA0006",
        "technique":   "OS Credential Dumping",
        "tech_id":     "T1003",
        "description": "Adversary dumps credentials from OS memory (e.g. LSASS via Mimikatz).",
        "url":         "https://attack.mitre.org/techniques/T1003/",
    },
    "SQL Injection": {
        "tactic":      "Initial Access",
        "tactic_id":   "TA0001",
        "technique":   "Exploit Public-Facing Application",
        "tech_id":     "T1190",
        "description": "Adversary exploits a vulnerability in a web-facing application.",
        "url":         "https://attack.mitre.org/techniques/T1190/",
    },
    "XSS": {
        "tactic":      "Execution",
        "tactic_id":   "TA0002",
        "technique":   "Exploit Client Execution: Reflected XSS",
        "tech_id":     "T1059",
        "description": "Adversary injects malicious scripts into a trusted web application.",
        "url":         "https://attack.mitre.org/techniques/T1059/",
    },
    "Unauthorized Access": {
        "tactic":      "Privilege Escalation",
        "tactic_id":   "TA0004",
        "technique":   "Valid Accounts",
        "tech_id":     "T1078",
        "description": "Adversary uses legitimate credentials to access systems without authorisation.",
        "url":         "https://attack.mitre.org/techniques/T1078/",
    },
    "DDoS": {
        "tactic":      "Impact",
        "tactic_id":   "TA0040",
        "technique":   "Network Denial of Service",
        "tech_id":     "T1498",
        "description": "Adversary overwhelms network resources to deny service to legitimate users.",
        "url":         "https://attack.mitre.org/techniques/T1498/",
    },
    "Intrusion": {
        "tactic":      "Discovery",
        "tactic_id":   "TA0007",
        "technique":   "Network Service Discovery",
        "tech_id":     "T1046",
        "description": "Adversary scans to enumerate services and find attack vectors.",
        "url":         "https://attack.mitre.org/techniques/T1046/",
    },
    # FIX 6: Added Lateral Movement
    "Lateral Movement": {
        "tactic":      "Lateral Movement",
        "tactic_id":   "TA0008",
        "technique":   "Remote Services",
        "tech_id":     "T1021",
        "description": "Adversary pivots across internal network using remote services.",
        "url":         "https://attack.mitre.org/techniques/T1021/",
    },
    # FIX 6: Added Privilege Escalation
    "Privilege Escalation": {
        "tactic":      "Privilege Escalation",
        "tactic_id":   "TA0004",
        "technique":   "Abuse Elevation Control Mechanism",
        "tech_id":     "T1548",
        "description": "Adversary circumvents access controls to gain elevated permissions.",
        "url":         "https://attack.mitre.org/techniques/T1548/",
    },
    # FIX 6: Added Data Exfiltration
    "Data Exfiltration": {
        "tactic":      "Exfiltration",
        "tactic_id":   "TA0010",
        "technique":   "Exfiltration Over C2 Channel",
        "tech_id":     "T1041",
        "description": "Adversary steals data by sending it out over the command-and-control channel.",
        "url":         "https://attack.mitre.org/techniques/T1041/",
    },
    # FIX 6: Added Port Scanning
    "Port Scanning": {
        "tactic":      "Reconnaissance",
        "tactic_id":   "TA0043",
        "technique":   "Active Scanning: Scanning IP Blocks",
        "tech_id":     "T1595.001",
        "description": "Adversary scans IP blocks and ports to map the target environment.",
        "url":         "https://attack.mitre.org/techniques/T1595/001/",
    },
}

_MITRE_ALIASES = {
    "sql injection":          "SQL Injection",
    "cross-site scripting":   "XSS",
    "cross site scripting":   "XSS",
    "brute force":            "Brute Force",
    "bruteforce":             "Brute Force",
    "credential stuffing":    "Credential Stuffing",
    "credential dumping":     "Credential Dumping",   # FIX 7
    "credential harvesting":  "Credential Dumping",   # FIX 7
    "lsass":                  "Credential Dumping",   # FIX 7
    "mimikatz":               "Credential Dumping",   # FIX 7
    "ddos":                   "DDoS",
    "distributed denial":     "DDoS",
    "ransomware":             "Ransomware",
    "malware":                "Malware",
    "phishing":               "Phishing",
    "unauthorized":           "Unauthorized Access",
    "unauthorized access":    "Unauthorized Access",
    "intrusion":              "Intrusion",
    "network intrusion":      "Intrusion",
    "lateral movement":       "Lateral Movement",     # FIX 7
    "privilege escalation":   "Privilege Escalation", # FIX 7
    "data exfiltration":      "Data Exfiltration",    # FIX 7
    "port scanning":          "Port Scanning",        # FIX 7
    "port scan":              "Port Scanning",        # FIX 7
}


def lookup_mitre(attack_type: str) -> str:
    if not attack_type or attack_type.strip().lower() in ("none", "", "unknown"):
        return "MITRE ATT&CK: No attack type detected — mapping skipped"

    normalized = attack_type.strip().lower()

    # Exact case-insensitive match against map keys
    for key, entry in _MITRE_MAP.items():
        if key.lower() == normalized:
            return _format_mitre(entry)

    # Exact alias match
    if normalized in _MITRE_ALIASES:
        canonical = _MITRE_ALIASES[normalized]
        if canonical in _MITRE_MAP:
            return _format_mitre(_MITRE_MAP[canonical])

    # Partial alias match (alias appears somewhere in the attack_type string)
    for alias, canonical in _MITRE_ALIASES.items():
        if alias in normalized and canonical in _MITRE_MAP:
            return _format_mitre(_MITRE_MAP[canonical])

    return (
        f"MITRE ATT&CK: No mapping found for '{attack_type}' — "
        f"check https://attack.mitre.org/techniques/ manually"
    )


def _format_mitre(entry: dict) -> str:
    return (
        f"MITRE ATT&CK | Tactic: {entry['tactic']} ({entry['tactic_id']}) | "
        f"Technique: {entry['technique']} ({entry['tech_id']}) | "
        f"{entry['description']} | Ref: {entry['url']}"
    )


# ── Tool 6: Domain Lookup ─────────────────────────────────────────────────────
def lookup_domain(domain: str) -> str:
    """
    FIX 8: Strip protocol, path, port numbers and trailing dots before DNS
           resolution. "evil.com:8080/path" or "evil.com." would raise
           socket.gaierror and silently return an error instead of a result.
    """
    domain = re.sub(r"https?://", "", domain)   # strip protocol
    domain = domain.split("/")[0]               # strip path
    domain = domain.split(":")[0]               # FIX 8: strip port
    domain = domain.rstrip(".")                 # FIX 8: strip trailing dot
    domain = domain.strip()

    if not domain:
        return "Invalid domain provided"
    try:
        ip  = socket.gethostbyname(domain)
        rep = check_ip_reputation(ip)
        return f"Domain '{domain}' → IP {ip} | {rep}"
    except socket.gaierror:
        return f"Could not resolve domain: {domain}"
    except Exception as e:
        return f"Domain lookup error: {str(e)[:60]}"


# ── Helper ────────────────────────────────────────────────────────────────────
def _is_valid_ip(ip: str) -> bool:
    """
    FIX 9: Reject leading-zero octets (e.g. '192.168.01.1') which are invalid
           IPv4 and cause some APIs to reject the request silently.
    """
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        for p in parts:
            # FIX 9: leading zeros are invalid
            if len(p) > 1 and p.startswith("0"):
                return False
            val = int(p)
            if not (0 <= val <= 255):
                return False
        return True
    except ValueError:
        return False