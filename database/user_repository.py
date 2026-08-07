import sqlite3

DB = "data/bridgebot.db"

def get_user(username):

    conn = sqlite3.connect(DB)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    user = cursor.fetchone()

    conn.close()

    return user