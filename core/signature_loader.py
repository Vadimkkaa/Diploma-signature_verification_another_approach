import cv2  # библиотека OpenCV — для работы с изображениями
import os   # для проверки, существует ли путь
import matplotlib.pyplot as plt  # для визуализации изображения (если нужно)


class SignatureLoader:
    """
    Класс для загрузки изображения с подписью.
    """

    def __init__(self, image_path):
        """
        Инициализация с указанием пути до изображения.
        """
        self.image_path = image_path
        self.image = None  # сюда загрузим изображение

    def load_image(self):
        """
        Загружает изображение в оттенках серого.
        """
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Файл не найден: {self.image_path}")

        # Загружаем изображение в режиме grayscale (0 означает ч/б)
        self.image = cv2.imread(self.image_path, 0)

        if self.image is None:
            raise ValueError("Ошибка при загрузке изображения!")

        return self.image

    def show_image(self):
        """
        Отображает загруженное изображение с помощью matplotlib.
        """
        if self.image is None:
            raise ValueError("Сначала нужно загрузить изображение!")

        plt.imshow(self.image, cmap='gray')  # показываем как ч/б
        plt.title("Изображение подписи")
        plt.axis('off')
        plt.show()
