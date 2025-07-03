import sqlite3

DB_PATH = "scores.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS scores (name TEXT PRIMARY KEY, points INTEGER)"
    )
    conn.commit()
    conn.close()


def add_user(name: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO scores (name, points) VALUES (?, 0)", (name,))
    conn.commit()
    conn.close()


def add_points(name: str, points: int = 1):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO scores (name, points) VALUES (?, 0)", (name,))
    c.execute("UPDATE scores SET points = points + ? WHERE name = ?", (points, name))
    conn.commit()
    conn.close()


def get_scores():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, points FROM scores ORDER BY points DESC")
    rows = c.fetchall()
    conn.close()
    return rows
