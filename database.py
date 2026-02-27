import sqlite3
import json
import numpy as np

DB_NAME = "members.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            membership_type TEXT DEFAULT 'Premium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_member(name, embedding, membership_type='Premium'):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Convert embedding (list of floats) to JSON string
    embedding_json = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
    cursor.execute('INSERT INTO members (name, embedding, membership_type) VALUES (?, ?, ?)', 
                   (name, embedding_json, membership_type))
    conn.commit()
    conn.close()

def get_all_members():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT name, embedding FROM members')
    rows = cursor.fetchall()
    conn.close()
    
    members = []
    for row in rows:
        members.append({
            "name": row[0],
            "embedding": json.loads(row[1])
        })
    return members

# Ensure database is initialized when module is imported
init_db()

if __name__ == "__main__":
    # When run as script, also print confirmation
    print("Database initialized.")
