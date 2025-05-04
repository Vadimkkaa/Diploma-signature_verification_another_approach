import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from core.signature_features import SignatureFeaturesExtractor

class SignatureComparator:
    """
    Сравнивает две подписи и возвращает вектор признаков сходства:
    [cosine_LBP, cosine_Curvature, cosine_HOG]
    """

    def __init__(self):
        pass

    def compare(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Parameters:
            image1, image2: нормализованные изображения (300x150)

        Returns:
            np.ndarray: вектор [cos_lbp, cos_curv, cos_hog]
        """
        extractor1 = SignatureFeaturesExtractor(image1)
        extractor2 = SignatureFeaturesExtractor(image2)

        # Извлекаем признаки
        lbp1 = extractor1.extract_lbp_features()
        lbp2 = extractor2.extract_lbp_features()

        curv1 = extractor1.extract_curvature_features()
        curv2 = extractor2.extract_curvature_features()

        hog1 = extractor1.extract_hog_features()
        hog2 = extractor2.extract_hog_features()

        # Нормализуем признаки (L2)
        lbp1 = normalize(lbp1.reshape(1, -1))[0]
        lbp2 = normalize(lbp2.reshape(1, -1))[0]

        curv1 = normalize(curv1.reshape(1, -1))[0]
        curv2 = normalize(curv2.reshape(1, -1))[0]

        hog1 = normalize(hog1.reshape(1, -1))[0]
        hog2 = normalize(hog2.reshape(1, -1))[0]

        # Вычисляем cosine similarity
        cos_lbp = cosine_similarity([lbp1], [lbp2])[0][0]
        cos_curv = cosine_similarity([curv1], [curv2])[0][0]
        cos_hog = cosine_similarity([hog1], [hog2])[0][0]

        return np.array([cos_lbp, cos_curv, cos_hog], dtype=np.float32)
