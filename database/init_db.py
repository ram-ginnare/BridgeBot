import sqlite3
import bcrypt

DB = "data/bridgebot.db"

conn = sqlite3.connect(DB)

cursor = conn.cursor()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS users(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT UNIQUE,
                       password_hash TEXT,
                       email TEXT,
                       role TEXT,
                       status INTEGER,
                       created_date TEXT,
                       last_login TEXT,
                       department TEXT,
                       team TEXT
               )
               """)

password = bcrypt.hashpw(
    "admin123".encode(),
    bcrypt.gensalt()
).decode()

# 1. Admin
cursor.execute(
    """
    INSERT OR IGNORE INTO users 
    (id, username, password_hash, email, role, status, created_date, last_login, department, team) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        1,
        "admin",
        password,
        "admin@bridgebot.com",
        "ADMIN",
        1,
        "2026-08-04 04:13:20",
        None,
        "AI",
        "BridgeBot",
    ),
)

# 2. Ram
cursor.execute(
    """
    INSERT OR IGNORE INTO users 
    (id, username, password_hash, email, role, status, created_date, last_login, department, team) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        2,
        "ram",
        password,
        "ram@bridgebot.com",
        "EDITOR",
        1,
        "2026-08-05 03:16:24",
        None,
        "AI",
        "BridgeBot",
    ),
)

# 3. Rakesh
cursor.execute(
    """
    INSERT OR IGNORE INTO users 
    (id, username, password_hash, email, role, status, created_date, last_login, department, team) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        3,
        "rakesh",
        password,
        "rakesh@bridgebot.com",
        "VIEWER",
        1,
        "2026-08-05 03:16:24",
        None,
        "AI",
        "BridgeBot_DB",
    ),
)

# 4. Bharat
cursor.execute(
    """
    INSERT OR IGNORE INTO users 
    (id, username, password_hash, email, role, status, created_date, last_login, department, team) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        4,
        "bharat",
        password,
        "bharat@bridgebot.com",
        "VIEWER",
        1,
        "2026-08-05 03:16:24",
        None,
        "AI",
        "AIML",
    ),
)

conn.commit()

conn.close()

print("Database Initialized")