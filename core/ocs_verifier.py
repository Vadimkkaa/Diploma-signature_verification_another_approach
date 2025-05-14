import os
import joblib
import numpy as np
from sklearn.svm import OneClassSVM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class OCSVMVerifier:
    """
    Класс для обучения и верификации на основе сравнения подписи с эталонными.
    обучение на парах, проверка через голосование.
    """

    def __init__(self, model_dir="storage/models"):
        self.model = None
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def fit_on_pairs(self, feature_vectors: list):
        """
        Обучает OC-SVM на признаках сходства между парами подлинных подписей.
        feature_vectors: список векторов сравнения [corr_lbp, corr_curv], один на каждую пару подлинных подписей.
        """
        self.model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.model.fit(feature_vectors)

    def verify_with_voting(self, new_signature_image, reference_images, comparator, threshold=0.75, return_metrics=False):
        """
        Сравнивает новую подпись с каждой из эталонных, формирует N сравнений,
        передаёт каждый вектор в модель и голосует.

        Parameters:
            new_signature_image: нормализованная подпись, которую проверяем
            reference_images: список нормализованных эталонных подписей
            comparator: экземпляр SignatureComparator
            threshold: доля голосов 'за', чтобы принять подпись (по умолчанию 0.8)

        Returns:
            int: 1 (принято как своя) или -1 (отклонено)
        """
        if self.model is None:
            raise ValueError("Модель не обучена или не загружена.")

        votes = []
        for ref_img in reference_images:
            feature_vector = comparator.compare(new_signature_image, ref_img)  # → [corr_lbp, corr_curv]
            result = self.model.predict([feature_vector])[0]
            votes.append(result)

        count_positive = sum(1 for v in votes if v == 1)
        if return_metrics:
            return (1 if count_positive >= int(len(votes) * threshold) else -1), {
                "votes_for": count_positive,
                "total": len(votes),
                "threshold": threshold
            }
        else:
            return 1 if count_positive >= int(len(votes) * threshold) else -1  # ≥75% голосов "за"

    def save_model(self, user_id):
        """
        Сохраняет обученную модель в файл (по ID пользователя).
        """

        if self.model is None:
            raise ValueError("Нет модели для сохранения.")

        path = os.path.join(BASE_DIR, "storage", "models", f"user_{user_id}_model.pkl")
        joblib.dump(self.model, path)

    def load_model(self, user_id):
        """
        Загружает модель из файла по ID пользователя.
        """

        path = os.path.join(self.model_dir, f"user_{user_id}_model.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель пользователя user_{user_id} не найдена.")

        self.model = joblib.load(path)
