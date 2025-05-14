import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.signature_normalizer import SignatureNormalizer
from core.signature_features import SignatureFeaturesExtractor
from core.ocs_verifier import OCSVMVerifier
from core.signature_comparator import SignatureComparator
from utils.results_logger import ResultsLogger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

originals_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "originals")
forgeries_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "forgeries")

user_id = "5"

original_files = sorted([
    f for f in os.listdir(originals_path)
    if f.startswith(f"original_{user_id}_")
])

forgery_files = sorted([
    f for f in os.listdir(forgeries_path)
    if f.startswith(f"forgeries_{user_id}_")
])

# Используем 15 оригинальных для обучения, 9 оставшихся — на тест
train_files = original_files[:15]
test_originals = original_files[15:]  # оставшиеся оригиналы
test_forgeries = forgery_files        # 24 подделки

# Нормализуем и сохраняем оригиналы для обучения
reference_images = []
for fname in train_files:
    img = cv2.imread(os.path.join(originals_path, fname), 0)
    norm = SignatureNormalizer(img).normalize()
    reference_images.append(norm)

# Формируем пары для обучения (C(10, 2) = 45)
comparator = SignatureComparator()
pairwise_vectors = []
for i in range(len(reference_images)):
    for j in range(i + 1, len(reference_images)):
        vec = comparator.compare(reference_images[i], reference_images[j])
        pairwise_vectors.append(vec)

# Обучаем модель
verifier = OCSVMVerifier()
verifier.fit_on_pairs(pairwise_vectors)
verifier.save_model(user_id)

# Тестирование
X_test = []
y_true = []
y_pred = []

# Проверка на оригинальных
for fname in test_originals:
    img = cv2.imread(os.path.join(originals_path, fname), 0)
    norm = SignatureNormalizer(img).normalize()
    result = verifier.verify_with_voting(norm, reference_images, comparator)
    y_pred.append(result)
    y_true.append(1)

# Проверка на подделках
for fname in test_forgeries:
    img = cv2.imread(os.path.join(forgeries_path, fname), 0)
    norm = SignatureNormalizer(img).normalize()
    result = verifier.verify_with_voting(norm, reference_images, comparator)
    y_pred.append(result)
    y_true.append(-1)

# Подсчёт метрик
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label=1)
rec = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)

tp = sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))
fp = sum((np.array(y_pred) == 1) & (np.array(y_true) == -1))
fn = sum((np.array(y_pred) == -1) & (np.array(y_true) == 1))
tn = sum((np.array(y_pred) == -1) & (np.array(y_true) == -1))

# Вывод
print("\n📊 Результаты тестирования модели OC-SVM для",user_id,"человека")
print(f"✅ Accuracy:  {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall:    {rec:.4f}")
print(f"✅ F1-score:  {f1:.4f}")

print(f"\n📌 TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

# Логируем
logger = ResultsLogger()
metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1
}
conf_matrix = {
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "tn": tn
}
logger.log(user_id, metrics, conf_matrix)
print("📄 Результаты сохранены в базу данных.")
