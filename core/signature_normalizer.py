import cv2
import numpy as np


class SignatureNormalizer:
    """
    Класс для нормализации изображения подписи.
    Приводит к стандартному размеру, центрирует и очищает от пустот.
    """

    def __init__(self, image, target_size=(300, 150)):
        """
        Принимает изображение и желаемый размер (по умолчанию 300×150).
        """
        self.original = image
        self.target_width, self.target_height = target_size
        self.normalized = None

    def remove_blank_edges(self, img):
        """
        Удаляет пустые (белые) края вокруг подписи.
        """
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y + h, x:x + w]
        return cropped

    def normalize(self):
        """
        Нормализует изображение: обрезка, масштаб, центрирование.
        """
        img = self.remove_blank_edges(self.original)

        # Масштабируем под размер, сохраняя пропорции
        h, w = img.shape
        scale = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Создаём холст и вставляем по центру
        canvas = np.ones((self.target_height, self.target_width), dtype=np.uint8) * 255
        x_offset = (self.target_width - new_w) // 2
        y_offset = (self.target_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        self.normalized = canvas
        return canvas

    def show(self):
        """
        Показывает нормализованную подпись.
        """
        if self.normalized is None:
            raise ValueError("Сначала вызовите normalize()")
        cv2.imshow("Нормализованная подпись", self.normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, path="storage/data/cropped/normalized_signature.png"):
        """
        Сохраняет нормализованную подпись в файл.
        """
        if self.normalized is None:
            raise ValueError("Сначала вызовите normalize()")
        cv2.imwrite(path, self.normalized)
