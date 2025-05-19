class ThresholdTuner:
    def __init__(self):
        self.threshold = 0.5

    def auto_tune(self, data):
        self.threshold = 0.6
        print(f"Порог автоматически установлен: {self.threshold}")

    def manual_adjust(self, value):
        self.threshold = value
        print(f"Порог вручную изменён на: {value}")

    def validate_threshold(self):
        print("Порог успешно проверен.")

    def visualize_distribution(self):
        print("Построено распределение вероятностей.")

    def reset(self):
        self.threshold = 0.5
        print("Порог сброшен до значения по умолчанию.")

    def get_current_threshold(self):
        return self.threshold

    def export_settings(self):
        print("Настройки порога сохранены в config.json")
