import cv2
import numpy as np
import os


class SignatureLocator:
    """
    Класс для поиска и выделения области подписи на изображении документа.
    """

    def __init__(self, image):
        """
        Принимает изображение (grayscale) при инициализации.
        """
        self.original_image = image
        self.signature_roi = None
        self.coordinates = None  # запомним координаты вырезанной области

    def preprocess(self):
        """
        Выполняет размытие и адаптивную бинаризацию изображения.
        """
        blurred = cv2.GaussianBlur(self.original_image, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 10
        )

        return thresh

    def locate_signature(self):
        """
        Ищет область подписи, вырезает её и возвращает.
        """
        thresh = self.preprocess()

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        image_height = self.original_image.shape[0]
        candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if y > image_height * 0.4 and w > 60 and h < 180:
                area = w * h
                candidates.append((area, x, y, w, h))


        if not candidates:
            raise ValueError("Область подписи не найдена.")

        candidates.sort(reverse=True)
        _, x, y, w, h = candidates[0]

        # Запоминаем координаты
        self.coordinates = (x, y, w, h)

        # Вырезаем подпись
        roi = self.original_image[y:y + h, x:x + w]

        # Обрезаем нижнюю часть (например, подпись текста “(звание, подпись)”)
        h_crop = int(roi.shape[0] * 0.85)
        roi = roi[:h_crop, :]

        # Дополнительно сглаживаем шумы
        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        # Морфология — фильтрация фона
        kernel = np.ones((2, 2), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        self.signature_roi = roi
        return roi

        # Удаляем возможную черную полоску справа (если есть)
        w_trim = roi.shape[1] - 10
        self.signature_roi = roi[:, :w_trim]

    def show_signature(self):
        """
        Отображает вырезанную подпись.
        """
        if self.signature_roi is None:
            raise ValueError("Сначала нужно выполнить locate_signature()")

        cv2.imshow("Вырезанная подпись (чистая)", self.signature_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_signature(self, path="storage/data/cropped/signature_demo.png"):
        """
        Сохраняет вырезанную подпись в указанный путь.
        """
        if self.signature_roi is None:
            raise ValueError("Сначала нужно выполнить locate_signature()")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, self.signature_roi)

    def show_highlighted_area(self):
        """
        Показывает оригинальное изображение с прямоугольником вокруг найденной подписи.
        """
        if self.coordinates is None:
            raise ValueError("Сначала нужно выполнить locate_signature()")

        x, y, w, h = self.coordinates
        img_color = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Область подписи на документе", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
