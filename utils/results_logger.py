import sqlite3
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class ResultsLogger:
    def __init__(self, db_path = os.path.join(BASE_DIR, "storage", "results.db")):
        self.db_path = db_path
        self._create_table()
        self._create_verification_table()

    def _create_table(self):
        """
        Создаёт таблицу результатов, если её ещё нет.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    tp INTEGER,
                    fp INTEGER,
                    fn INTEGER,
                    tn INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def _create_verification_table(self):
        """
        Создаёт таблицу verification_logs, если её нет.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS verification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    result INTEGER,
                    votes_for INTEGER,
                    threshold REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            ''')
            conn.commit()

    def log_verification(self, user_id, result, votes_for, threshold):
        """
        Записывает факт верификации в таблицу verification_logs.
        """
        from datetime import datetime
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO verification_logs (user_id, result, votes_for, threshold, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                result,
                votes_for,
                threshold,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()

    def get_verification_logs(self):
        """
        Возвращает все записи из таблицы verification_logs.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, user_id, result, votes_for, threshold, timestamp
                FROM verification_logs
                ORDER BY timestamp DESC
            ''')
            return cursor.fetchall()

    def log(self, user_id, metrics: dict, confusion: dict):
        """
        Записывает метрики и confusion matrix в таблицу.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO test_results (user_id, accuracy, precision, recall, f1_score, tp, fp, fn, tn)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                int(confusion['tp']),
                int(confusion['fp']),
                int(confusion['fn']),
                int(confusion['tn'])
            ))
            conn.commit()

    def view_results(self):
        """
        Показывает все строки из таблицы результатов.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_id, accuracy, precision, recall, f1_score, tp, fp, fn, tn, timestamp FROM test_results")
            rows = cursor.fetchall()
            aver_acc=0

            if not rows:
                print("⚠️ База пуста.")
                return

            print("\n📊 Сохранённые результаты:")
            print("-" * 90)
            for row in rows:
                log_id, user_id, acc, prec, rec, f1, tp, fp, fn, tn, timestamp = row
                print(
                    f"🧾 [#{log_id}] User: {user_id} | "
                    f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}\n"
                    f"      → TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn} | ⏱ {timestamp}\n"
                    + "-" * 90
                )

    def delete_all(self):
        """
        Полностью очищает таблицу результатов и сбрасывает счётчик автоинкремента ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM test_results")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='test_results'")
            conn.commit()
            print("🧹 Все записи удалены, счётчик ID сброшен.")

    def delete_verification_by_user(self, user_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM verification_logs WHERE user_id = ?", (user_id,))
            conn.commit()
            print(f"🗑️ Удалены все проверки для user_id = {user_id}")
