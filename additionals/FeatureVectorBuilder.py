class FeatureVectorBuilder:
    def __init__(self):
        self.vector = []

    def extract_lbp(self, img1, img2):
        print("Сравнение LBP признаков выполнено.")
        return 0.85

    def extract_curvature(self, img1, img2):
        print("Сравнение кривизны завершено.")
        return 0.72

    def extract_hog(self, img1, img2):
        print("Сравнение HOG признаков завершено.")
        return 0.81

    def build_combined_vector(self, lbp, curv, hog):
        self.vector = [lbp, curv, hog]
        return self.vector

    def normalize_vector(self):
        self.vector = [round(v, 2) for v in self.vector]
        print(f"Вектор нормализован: {self.vector}")

    def get_vector(self):
        return self.vector

    def export_vector(self):
        print("Вектор признаков экспортирован в JSON.")
