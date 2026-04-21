"""
tests/test_tools.py — Full test suite for tools.py
Run: pytest tests/test_tools.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# ── Make sure tools.py is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import detect_anomaly, check_ip_reputation, check_virustotal, lookup_cve, lookup_domain, _is_valid_ip


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1: detect_anomaly
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectAnomaly:

    # ── Critical patterns ─────────────────────────────────────────────────────

    def test_sql_injection_or_1_equals_1(self):
        result = detect_anomaly("User input contains ' OR 1=1 -- in the login form")
        assert result.startswith("Critical")
        assert "SQL" in result

    def test_sql_injection_union_select(self):
        result = detect_anomaly("Detected UNION SELECT * FROM users in query string")
        assert result.startswith("Critical")

    def test_sql_drop_table(self):
        result = detect_anomaly("Query contains DROP TABLE users;")
        assert result.startswith("Critical")

    def test_xss_script_tag(self):
        result = detect_anomaly("Input field contains <script>alert('xss')</script>")
        assert result.startswith("Critical")

    def test_xss_label(self):
        result = detect_anomaly("Cross-site scripting attempt detected in form submission")
        assert result.startswith("Critical")

    def test_ransomware(self):
        result = detect_anomaly("Ransomware detected encrypting files on C: drive")
        assert result.startswith("Critical")

    def test_malware(self):
        result = detect_anomaly("Malware detected on endpoint 192.168.1.5")
        assert result.startswith("Critical")

    def test_trojan(self):
        result = detect_anomaly("Trojan horse found in downloaded executable")
        assert result.startswith("Critical")

    def test_rootkit(self):
        result = detect_anomaly("Rootkit activity detected in kernel space")
        assert result.startswith("Critical")

    def test_unauthorized_access(self):
        result = detect_anomaly("Unauthorized access attempt detected in admin panel")
        assert result.startswith("Critical")

    def test_brute_force_label(self):
        result = detect_anomaly("Brute force attack detected from IP 10.0.0.5")
        assert result.startswith("Critical")

    def test_phishing(self):
        result = detect_anomaly("User received a phishing email from unknown sender")
        assert result.startswith("Critical")

    def test_credential_dumping(self):
        result = detect_anomaly("Credential dumping attempt detected via LSASS")
        assert result.startswith("Critical")

    def test_mimikatz(self):
        result = detect_anomaly("mimikatz tool found running on workstation")
        assert result.startswith("Critical")

    def test_privilege_escalation(self):
        result = detect_anomaly("privilege escalation attempt detected for user guest")
        assert result.startswith("Critical")

    # ── Suspicious patterns ───────────────────────────────────────────────────

    def test_multiple_failed_logins(self):
        result = detect_anomaly("Multiple failed login attempts from IP 192.168.1.10")
        assert result.startswith("Suspicious")

    def test_ddos_traffic_spike(self):
        result = detect_anomaly("Sudden spike in traffic causing server slowdown")
        assert result.startswith("Suspicious")

    def test_ddos_label(self):
        result = detect_anomaly("DDoS attack suspected on port 80")
        assert result.startswith("Suspicious")

    def test_port_scan(self):
        result = detect_anomaly("Port scan detected from external IP address")
        assert result.startswith("Suspicious")

    def test_bulk_download(self):
        result = detect_anomaly("User performed bulk download of 50,000 customer records")
        assert result.startswith("Suspicious")

    def test_after_hours_access(self):
        result = detect_anomaly("Admin logged in at 3am outside normal working hours")
        assert result.startswith("Suspicious")

    def test_unusual_outbound(self):
        result = detect_anomaly("Unusual outbound data transfer to unknown server")
        assert result.startswith("Suspicious")

    def test_lateral_movement(self):
        result = detect_anomaly("Lateral movement detected across internal network segments")
        assert result.startswith("Suspicious")

    # ── Normal / Safe ─────────────────────────────────────────────────────────

    def test_normal_login(self):
        result = detect_anomaly("User logged in successfully from registered device")
        assert result.startswith("Normal")

    def test_normal_traffic(self):
        result = detect_anomaly("Server received normal traffic from users during peak hours")
        assert result.startswith("Normal")

    def test_empty_log(self):
        result = detect_anomaly("")
        assert result.startswith("Normal")

    def test_routine_backup(self):
        result = detect_anomaly("Scheduled backup completed successfully at 02:00")
        assert result.startswith("Normal")

    # ── Case insensitivity ────────────────────────────────────────────────────

    def test_case_insensitive_critical(self):
        result = detect_anomaly("RANSOMWARE DETECTED ON SYSTEM")
        assert result.startswith("Critical")

    def test_case_insensitive_suspicious(self):
        result = detect_anomaly("MULTIPLE FAILED LOGIN ATTEMPTS DETECTED")
        assert result.startswith("Suspicious")

    # ── Output format ─────────────────────────────────────────────────────────

    def test_output_uses_pipe_separator(self):
        result = detect_anomaly("SQL injection found in query")
        assert "|" in result

    def test_normal_output_format(self):
        result = detect_anomaly("everything is fine today")
        assert result == "Normal | No known threat patterns detected"


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: check_ip_reputation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckIpReputation:

    # ── Local fallback (no API key) ───────────────────────────────────────────

    def test_known_malicious_ip_fallback(self):
        with patch("tools.ABUSEIPDB_API_KEY", ""):
            result = check_ip_reputation("192.168.1.10")
        assert "Malicious" in result

    def test_known_safe_ip_fallback(self):
        with patch("tools.ABUSEIPDB_API_KEY", ""):
            result = check_ip_reputation("8.8.8.8")
        assert "Clean" in result

    def test_unknown_ip_fallback(self):
        with patch("tools.ABUSEIPDB_API_KEY", ""):
            result = check_ip_reputation("203.0.113.42")
        assert "Unknown" in result

    # ── Invalid IP ────────────────────────────────────────────────────────────

    def test_invalid_ip_format(self):
        result = check_ip_reputation("not_an_ip")
        assert "Invalid" in result

    def test_invalid_ip_out_of_range(self):
        result = check_ip_reputation("999.999.999.999")
        assert "Invalid" in result

    def test_empty_ip(self):
        result = check_ip_reputation("")
        assert "Invalid" in result

    # ── AbuseIPDB API mocked ──────────────────────────────────────────────────

    def test_abuseipdb_malicious_score(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {
            "abuseConfidenceScore": 95,
            "isp": "Bad ISP",
            "countryCode": "RU",
            "totalReports": 150
        }}
        with patch("tools.ABUSEIPDB_API_KEY", "fake_key"), \
             patch("tools.requests.get", return_value=mock_resp):
            result = check_ip_reputation("45.33.32.156")
        assert "Malicious" in result
        assert "95" in result

    def test_abuseipdb_suspicious_score(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {
            "abuseConfidenceScore": 25,
            "isp": "Some ISP",
            "countryCode": "CN",
            "totalReports": 10
        }}
        with patch("tools.ABUSEIPDB_API_KEY", "fake_key"), \
             patch("tools.requests.get", return_value=mock_resp):
            result = check_ip_reputation("1.2.3.4")
        assert "Suspicious" in result

    def test_abuseipdb_clean_score(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {
            "abuseConfidenceScore": 0,
            "isp": "Google LLC",
            "countryCode": "US",
            "totalReports": 0
        }}
        with patch("tools.ABUSEIPDB_API_KEY", "fake_key"), \
             patch("tools.requests.get", return_value=mock_resp):
            result = check_ip_reputation("8.8.8.8")
        assert "Clean" in result

    def test_abuseipdb_timeout_returns_unknown(self):
        import requests as req
        with patch("tools.ABUSEIPDB_API_KEY", "fake_key"), \
             patch("tools.requests.get", side_effect=req.exceptions.Timeout):
            result = check_ip_reputation("1.2.3.4")
        assert "Unknown" in result or "timeout" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3: check_virustotal
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckVirustotal:

    def test_no_api_key_returns_message(self):
        with patch("tools.VIRUSTOTAL_API_KEY", ""):
            result = check_virustotal("1.2.3.4")
        assert "not set" in result.lower() or "API key" in result

    def test_invalid_ip(self):
        with patch("tools.VIRUSTOTAL_API_KEY", "fake_key"):
            result = check_virustotal("bad_ip")
        assert "Invalid" in result

    def test_malicious_result(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"attributes": {"last_analysis_stats": {
            "malicious": 10, "suspicious": 2, "harmless": 50
        }}}}
        with patch("tools.VIRUSTOTAL_API_KEY", "fake_key"), \
             patch("tools.requests.get", return_value=mock_resp):
            result = check_virustotal("1.2.3.4")
        assert "Malicious" in result
        assert "10" in result

    def test_suspicious_result(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"attributes": {"last_analysis_stats": {
            "malicious": 0, "suspicious": 3, "harmless": 60
        }}}}
        with patch("tools.VIRUSTOTAL_API_KEY", "fake_key"), \
             patch("tools.requests.get", return_value=mock_resp):
            result = check_virustotal("1.2.3.4")
        assert "Suspicious" in result

    def test_clean_result(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"attributes": {"last_analysis_stats": {
            "malicious": 0, "suspicious": 0, "harmless": 70
        }}}}
        with patch("tools.VIRUSTOTAL_API_KEY", "fake_key"), \
             patch("tools.requests.get", return_value=mock_resp):
            result = check_virustotal("8.8.8.8")
        assert "Clean" in result

    def test_api_error_handled(self):
        with patch("tools.VIRUSTOTAL_API_KEY", "fake_key"), \
             patch("tools.requests.get", side_effect=Exception("connection error")):
            result = check_virustotal("1.2.3.4")
        assert "failed" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4: lookup_cve
# ═══════════════════════════════════════════════════════════════════════════════

class TestLookupCve:

    def test_no_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        with patch("tools.requests.get", return_value=mock_resp):
            result = lookup_cve("zzznonexistentkeyword999")
        assert "No CVEs found" in result

    def test_returns_cve_ids(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [
            {"cve": {
                "id": "CVE-2023-1234",
                "descriptions": [{"value": "A critical SQL injection vulnerability."}],
                "metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "CRITICAL"}}]}
            }}
        ]}
        with patch("tools.requests.get", return_value=mock_resp):
            result = lookup_cve("sql injection")
        assert "CVE-2023-1234" in result
        assert "CRITICAL" in result

    def test_api_failure_handled(self):
        with patch("tools.requests.get", side_effect=Exception("network error")):
            result = lookup_cve("ransomware")
        assert "failed" in result.lower()

    def test_keyword_passed_to_api(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        with patch("tools.requests.get", return_value=mock_resp) as mock_get:
            lookup_cve("log4shell")
            call_params = mock_get.call_args
        assert "log4shell" in str(call_params)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5: lookup_domain
# ═══════════════════════════════════════════════════════════════════════════════

class TestLookupDomain:

    def test_valid_domain_resolves(self):
        with patch("tools.socket.gethostbyname", return_value="93.184.216.34"), \
             patch("tools.check_ip_reputation", return_value="Clean IP — score 0/100"):
            result = lookup_domain("example.com")
        assert "example.com" in result
        assert "93.184.216.34" in result

    def test_strips_https(self):
        with patch("tools.socket.gethostbyname", return_value="1.2.3.4"), \
             patch("tools.check_ip_reputation", return_value="Clean IP"):
            result = lookup_domain("https://example.com/path")
        assert "example.com" in result

    def test_unresolvable_domain(self):
        import socket
        with patch("tools.socket.gethostbyname", side_effect=socket.gaierror):
            result = lookup_domain("notareal.domain.xyz")
        assert "Could not resolve" in result

    def test_empty_domain(self):
        result = lookup_domain("")
        assert "Invalid" in result


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: _is_valid_ip
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsValidIp:

    def test_valid_ips(self):
        assert _is_valid_ip("192.168.1.1")   is True
        assert _is_valid_ip("0.0.0.0")        is True
        assert _is_valid_ip("255.255.255.255") is True
        assert _is_valid_ip("8.8.8.8")        is True

    def test_invalid_ips(self):
        assert _is_valid_ip("256.0.0.1")    is False
        assert _is_valid_ip("192.168.1")    is False
        assert _is_valid_ip("not_an_ip")    is False
        assert _is_valid_ip("")             is False
        assert _is_valid_ip("192.168.1.1.1") is False
        assert _is_valid_ip("192.168.-1.1")  is False


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: detect_anomaly + check_ip_reputation together
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_critical_log_with_known_malicious_ip(self):
        log_text = "Brute force attack detected from IP 192.168.1.10"
        anomaly  = detect_anomaly(log_text)
        with patch("tools.ABUSEIPDB_API_KEY", ""):
            ip_rep = check_ip_reputation("192.168.1.10")
        assert anomaly.startswith("Critical")
        assert "Malicious" in ip_rep

    def test_normal_log_with_safe_ip(self):
        log_text = "User logged in successfully"
        anomaly  = detect_anomaly(log_text)
        with patch("tools.ABUSEIPDB_API_KEY", ""):
            ip_rep = check_ip_reputation("8.8.8.8")
        assert anomaly.startswith("Normal")
        assert "Clean" in ip_rep

    def test_suspicious_log_with_unknown_ip(self):
        log_text = "Multiple failed login attempts from IP 203.0.113.99"
        anomaly  = detect_anomaly(log_text)
        with patch("tools.ABUSEIPDB_API_KEY", ""):
            ip_rep = check_ip_reputation("203.0.113.99")
        assert anomaly.startswith("Suspicious")
        assert "Unknown" in ip_rep