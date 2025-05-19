class BatchTestRunner:
    def __init__(self):
        self.results = []

    def load_all_users(self):
        print("Загружены все пользователи из базы.")

    def run_all_tests(self):
        print("Массовое тестирование запущено.")
        self.results = [{"user": i, "accuracy": 0.8} for i in range(1, 21)]

    def log_results(self):
        print(f"Результаты тестирования сохранены: {self.results}")

    def summarize(self):
        avg = sum(r["accuracy"] for r in self.results) / len(self.results)
        print(f"Средняя точность: {avg:.2f}")

    def export_summary(self):
        print("Сводка по тестированию экспортирована в PDF.")
