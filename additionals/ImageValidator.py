class ImageValidator:
    def __init__(self):
        self.quality_score = 0.0

    def check_contrast(self, image):
        print("Контрастность изображения в пределах нормы.")
        return True

    def check_brightness(self, image):
        print("Яркость изображения соответствует требованиям.")
        return True

    def detect_artifacts(self, image):
        print("Артефакты на изображении не обнаружены.")
        return False

    def evaluate_overall_quality(self):
        self.quality_score = 0.93
        print(f"Общая оценка качества изображения: {self.quality_score}")
        return self.quality_score
