import sqlite3
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class ConfigManager:
    def __init__(self, db_path=os.path.join(BASE_DIR, "storage", "results.db")):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    language TEXT,
                    theme TEXT
                )
            ''')
            conn.commit()

    def load_config(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT language, theme FROM configs LIMIT 1")
            row = cursor.fetchone()
            if row:
                return {"language": row[0], "theme": row[1]}
            else:
                # Если пусто — создаём запись по умолчанию
                self.save_config(language="Русский", theme="Светлая")
                return {"language": "Русский", "theme": "Светлая"}

    def save_config(self, language, theme):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM configs")
            cursor.execute(
                "INSERT INTO configs (language, theme) VALUES (?, ?)",
                (language, theme)
            )
            conn.commit()
