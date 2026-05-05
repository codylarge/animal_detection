import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database.db")
print(f"DB_PATH: {os.path.abspath(DB_PATH)}")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_path TEXT,
            timestamp TEXT,
            top_species TEXT,
            total_hits INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_event(folder_path, timestamp, top_species, total_hits):
    conn = get_db()
    conn.execute(
        "INSERT INTO events (folder_path, timestamp, top_species, total_hits) VALUES (?, ?, ?, ?)",
        (folder_path, timestamp, top_species, total_hits)
    )
    conn.commit()
    conn.close()