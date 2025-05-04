import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern

class SignatureFeaturesExtractor:
    def __init__(self, image: np.ndarray):
        self.image = image

    def extract_lbp_features(self):
        """
        Вычисляет LBP только по граничным пикселям подписи.
        Использует uniform шаблоны (10 значений).
        """
        # Шаг 1: Бинаризация (обратная — подпись белая, фон чёрный)
        _, binary = cv2.threshold(
            self.image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Шаг 2: Находим контурные (граничные) пиксели
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        boundary = cv2.subtract(binary, eroded)

        # Шаг 3: LBP по этим граничным пикселям
        lbp_image = local_binary_pattern(self.image, P=8, R=1, method="uniform")

        # Маска только по граничным пикселям
        boundary_mask = boundary.astype(bool)
        lbp_values = lbp_image[boundary_mask]

        # Строим гистограмму по 10 uniform шаблонам
        hist, _ = np.histogram(lbp_values, bins=np.arange(0, 11), density=True)
        return hist.astype(np.float32)

    def extract_curvature_features(self):
        """
        Вычисляет кривизну на изображении и возвращает гистограмму из 40 интервалов.
        """
        # Сглаживаем изображение
        smoothed = gaussian_filter(self.image.astype(np.float32), sigma=1)

        # Градиенты первого и второго порядка
        gx, gy = np.gradient(smoothed)
        gxx, _ = np.gradient(gx)
        _, gyy = np.gradient(gy)

        # Кривизна = сумма вторых производных
        curvature = np.abs(gxx + gyy)

        # Обрезаем экстремальные значения
        curvature = np.clip(curvature, 0, 1.0007)

        # Строим гистограмму: 40 интервалов от 0 до 1.0007
        hist, _ = np.histogram(curvature, bins=40, range=(0, 1.0007), density=True)
        return hist.astype(np.float32)

    def extract_hog_features(self, cell_size=(30, 30), bins=8):
        """
        Делит изображение на ячейки (например, 10x5),
        вычисляет гистограммы градиентов в каждой ячейке (HOG).
        Возвращает объединённый и нормализованный вектор.
        """
        from skimage.feature import hog

        # Нормализуем изображение в диапазон 0-1 для hog
        image = self.image.astype(np.float32) / 255.0

        # Вычисляем HOG
        features = hog(
            image,
            orientations=bins,
            pixels_per_cell=cell_size,
            cells_per_block=(1, 1),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        # Возвращаем нормализованный вектор (делим на сумму для устойчивости)
        if np.sum(features) > 0:
            features /= np.sum(features)

        return features.astype(np.float32)


