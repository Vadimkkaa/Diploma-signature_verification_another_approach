import cv2
import numpy as np

class ImagePreprocessor:
    """
    Класс для предобработки изображения подписи перед извлечением признаков.
    """

    def __init__(self, image):
        """
        Принимает на вход изображение (уже загруженное, в формате NumPy массива).
        """
        self.original_image = image
        self.processed_image = None

    def apply_threshold(self):
        """
        Применяет бинаризацию по методу Отсу.
        """
        _, binary = cv2.threshold(self.original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.processed_image = binary
        return binary

    def remove_noise(self, kernel_size=(2, 2)):
        """
        Удаляет мелкие шумы с помощью морфологической фильтрации.
        """
        if self.processed_image is None:
            raise ValueError("Сначала выполните бинаризацию!")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        denoised = cv2.morphologyEx(self.processed_image, cv2.MORPH_OPEN, kernel)
        self.processed_image = denoised
        return denoised

    def get_result(self):
        """
        Возвращает итоговое обработанное изображение.
        """
        if self.processed_image is None:
            raise ValueError("Обработка ещё не выполнена!")
        return self.processed_image
