"""
memory_store.py — Persistent conversation memory using SQLite.

Unlike MemorySaver (which resets on restart), this stores all conversations
to disk so users can return to previous sessions.

FIXES APPLIED:
  - FIX 1: SQLite connection created with isolation_level=None (autocommit)
           to prevent "database is locked" errors when multiple Streamlit
           reruns hit the DB concurrently. Explicit conn.commit() calls kept
           for clarity but are now no-ops under autocommit.
  - FIX 2: load_history() ORDER BY was DESC then reversed in Python — correct
           behaviour but inefficient. Simplified to ASC with a Python-side
           limit slice.
  - FIX 3: get_all_threats() now also returns the tool_output field so the
           Threat Log tab can optionally display tool intelligence details.
  - FIX 4: Added close() method so callers (e.g. tests) can cleanly release
           the connection.
  - FIX 5: os.makedirs() was passed dirname of DB_PATH which returns "" for
           a bare filename like "memory.db", causing makedirs("") to fail on
           some platforms. Added a guard to skip makedirs when dirname is "".
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Dict
from config import DB_PATH
from logger import get_logger

log = get_logger(__name__)


class PersistentMemory:
    # AFTER
    def __init__(self, db_path: str = DB_PATH):
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()
        self._create_tables()
        log.info(f"Memory store initialised at {db_path}")
    
    def _conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
            )
        return self._local.conn
    
    def _create_tables(self):
        self._conn().executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                decision    TEXT    DEFAULT '',
                faithfulness REAL   DEFAULT 0.0,
                timestamp   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_thread ON conversations(thread_id);

            CREATE TABLE IF NOT EXISTS threat_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT,
                input       TEXT,
                decision    TEXT,
                attack_type TEXT,
                tool_output TEXT,
                faithfulness REAL,
                timestamp   TEXT
            );
        """)

    def save_message(self, thread_id: str, role: str, content: str,
                     decision: str = "", faithfulness: float = 0.0):
        self._conn().execute(
            "INSERT INTO conversations (thread_id,role,content,decision,faithfulness,timestamp) "
            "VALUES (?,?,?,?,?,?)",
            (thread_id, role, content, decision, faithfulness,
             datetime.now().isoformat())
        )

    def load_history(self, thread_id: str, limit: int = 6) -> List[Dict]:
        # FIX 2: query in ASC order directly — simpler and avoids double reversal
        rows = self._conn().execute(
            "SELECT role, content FROM conversations "
            "WHERE thread_id=? ORDER BY timestamp ASC",
            (thread_id,)
        ).fetchall()
        # Return only the last `limit` messages
        return [{"role": r, "content": c} for r, c in rows[-limit:]]

    def log_threat(self, thread_id: str, input_text: str, decision: str,
                   attack_type: str, tool_output: str, faithfulness: float):
        self._conn().execute(
            "INSERT INTO threat_log "
            "(thread_id,input,decision,attack_type,tool_output,faithfulness,timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            (thread_id, input_text, decision, attack_type,
             tool_output, faithfulness, datetime.now().isoformat())
        )

    def get_session_stats(self, thread_id: str) -> Dict:
        rows = self._conn().execute(
            "SELECT decision, COUNT(*) FROM threat_log "
            "WHERE thread_id=? GROUP BY decision",
            (thread_id,)
        ).fetchall()
        stats = {"threat": 0, "suspicious": 0, "safe": 0}
        for decision, count in rows:
            if decision in stats:
                stats[decision] = count
        return stats

    def get_all_threats(self, thread_id: str) -> List[Dict]:
        # FIX 3: also return tool_output for richer Threat Log display
        rows = self._conn().execute(
            "SELECT input, attack_type, tool_output, timestamp FROM threat_log "
            "WHERE thread_id=? AND decision='threat' ORDER BY timestamp DESC",
            (thread_id,)
        ).fetchall()
        return [
            {
                "input":       r[0],
                "attack_type": r[1],
                "tool_output": r[2],
                "timestamp":   r[3],
            }
            for r in rows
        ]

    def close(self):
        try:
            if hasattr(self._local, "conn"):
                self._local.conn.close()
        except Exception:
            pass


# Singleton instance shared across the app
_memory_store = None

def get_memory_store() -> PersistentMemory:
    global _memory_store
    if _memory_store is None:
        _memory_store = PersistentMemory()
    return _memory_store