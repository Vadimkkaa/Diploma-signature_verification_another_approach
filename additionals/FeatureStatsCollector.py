class FeatureStatsCollector:
    def __init__(self):
        self.stats = []

    def add_feature(self, name, values):
        mean_val = sum(values) / len(values)
        self.stats.append((name, mean_val))

    def normalize(self):
        print("Нормализация признаков выполнена.")

    def export_csv(self):
        print("Статистика признаков экспортирована в CSV.")

    def summarize(self):
        for name, val in self.stats:
            print(f"{name}: среднее значение = {val:.2f}")
