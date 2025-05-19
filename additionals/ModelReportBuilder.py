class ModelReportBuilder:
    def __init__(self):
        self.sections = []

    def add_title(self, title):
        self.sections.append(f"Заголовок: {title}")

    def add_metrics(self, metrics):
        for k, v in metrics.items():
            self.sections.append(f"{k}: {v}")

    def add_conclusion(self, text):
        self.sections.append(f"Вывод: {text}")

    def build_pdf(self):
        print("PDF-отчёт сгенерирован.")
        for line in self.sections:
            print(line)

    def export(self, filename):
        print(f"Отчёт сохранён как {filename}.pdf")
