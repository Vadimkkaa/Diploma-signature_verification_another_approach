import sqlite3
import os
import random
from datetime import datetime
from faker import Faker

# Абсолютный путь к корню проекта
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class UserRegistry:
    def __init__(self):
        self.db_path = os.path.join(BASE_DIR, "storage", "results.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _create_table(self):
        """
        Создаёт таблицу users, если она не существует.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    last_name TEXT,
                    first_name TEXT,
                    middle_name TEXT,
                    gender TEXT,
                    birth_date TEXT
                )
            ''')
            conn.commit()

    def add_user(self, last_name, first_name, middle_name, gender, birth_date):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (
                    last_name, first_name, middle_name, gender, birth_date
                ) VALUES (?, ?, ?, ?, ?)
            ''', (last_name, first_name, middle_name, gender, birth_date))
            conn.commit()

    def get_user(self, user_id):
        """
        Возвращает информацию о пользователе по ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            return cursor.fetchone()

    def list_users(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
            return rows

    def generate_users(self, count=20):
        fake = Faker("ru_RU")
        for _ in range(count):
            gender = random.choice(["М", "Ж"])
            if gender == "М":
                first = fake.first_name_male()
                last = fake.last_name_male()
                middle = fake.middle_name_male()
            else:
                first = fake.first_name_female()
                last = fake.last_name_female()
                middle = fake.middle_name_female()

            birth_date = fake.date_of_birth(minimum_age=18, maximum_age=60).strftime("%Y-%m-%d")
            self.add_user(last, first, middle, gender, birth_date)

        print(f"✅ Сгенерировано {count} пользователей.")

    def get_all_users(self):
        """
        Возвращает список словарей с полными данными пользователей.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, last_name, first_name, middle_name FROM users")
            rows = cursor.fetchall()
            return [
                {
                    "user_id": row[0],
                    "last_name": row[1],
                    "first_name": row[2],
                    "middle_name": row[3]
                }
                for row in rows
            ]

    def clear_users(self, reset_ids=True):
        """
        Удаляет всех пользователей из таблицы users.
        Если reset_ids=True, сбрасывает автоинкремент user_id.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users")
            if reset_ids:
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='users'")
            conn.commit()
        print("🧹 Все пользователи удалены.")
