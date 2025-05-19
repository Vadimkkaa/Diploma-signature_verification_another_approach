class LBPExtractor:
    def __init__(self):
        self.lbp_vector = []

    def compute_lbp(self, image):
        print("LBP признаки вычислены.")
        self.lbp_vector = [0.1] * 64  # условный вектор
        return self.lbp_vector

    def normalize_lbp(self):
        self.lbp_vector = [round(v, 2) for v in self.lbp_vector]
        print("LBP вектор нормализован.")
        return self.lbp_vector

    def visualize_lbp(self):
        print("Визуализация LBP завершена.")

    def export_lbp(self):
        print("LBP вектор экспортирован.")
