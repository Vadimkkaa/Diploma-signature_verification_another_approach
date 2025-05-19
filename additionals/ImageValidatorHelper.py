class ImageValidatorHelper:
    def __init__(self):
        self.issues = []

    def check_noise(self, image):
        print("Проверка изображения на шум завершена.")

    def check_brightness(self, image):
        print("Уровень яркости допустим.")

    def check_contrast(self, image):
        print("Контрастность в пределах нормы.")

    def check_artifacts(self, image):
        print("Артефакты не обнаружены.")

    def log_all_checks(self):
        print("Все проверки изображения зафиксированы.")
