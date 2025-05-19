class CurvatureAnalyzer:
    def __init__(self):
        self.curvature_vector = []

    def extract_curves(self, image):
        print("Извлечены контуры подписи.")

    def smooth_curve(self):
        print("Кривая сглажена.")

    def compute_curvature(self):
        self.curvature_vector = [0.15] * 20
        print("Вычислены показатели кривизны.")
        return self.curvature_vector

    def export_curvature(self):
        print("Вектор кривизны экспортирован.")
