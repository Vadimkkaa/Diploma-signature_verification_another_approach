import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.signature_normalizer import SignatureNormalizer
from core.signature_features import SignatureFeaturesExtractor
from core.signature_comparator import SignatureComparator
from core.ocs_verifier import OCSVMVerifier
from utils.results_logger import ResultsLogger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
originals_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "originals")
forgeries_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "forgeries")

comparator = SignatureComparator()
logger = ResultsLogger()

# Хранилище метрик
all_metrics = []

for user_id in range(1, 21):
    print(f"\n🔁 Тестируем модель OC-SVM для пользователя {user_id}...\n")

    original_files = sorted([
        f for f in os.listdir(originals_path)
        if f.startswith(f"original_{user_id}_")
    ])

    forgery_files = sorted([
        f for f in os.listdir(forgeries_path)
        if f.startswith(f"forgeries_{user_id}_")
    ])

    if len(original_files) < 24 or len(forgery_files) < 24:
        print(f"⚠️ Недостаточно данных для пользователя {user_id}. Пропускаем.\n")
        continue

    train_files = original_files[:15]
    test_originals = original_files[15:]
    test_forgeries = forgery_files

    # Подготовка эталонов
    reference_images = []
    for fname in train_files:
        img = cv2.imread(os.path.join(originals_path, fname), 0)
        norm = SignatureNormalizer(img).normalize()
        reference_images.append(norm)

    # Пары для обучения
    pairwise_vectors = []
    for i in range(len(reference_images)):
        for j in range(i + 1, len(reference_images)):
            vec = comparator.compare(reference_images[i], reference_images[j])
            pairwise_vectors.append(vec)

    # Обучение
    verifier = OCSVMVerifier()
    verifier.fit_on_pairs(pairwise_vectors)
    verifier.save_model(user_id)

    y_true, y_pred = [], []

    # Проверка оригиналов
    for fname in test_originals:
        img = cv2.imread(os.path.join(originals_path, fname), 0)
        norm = SignatureNormalizer(img).normalize()
        result = verifier.verify_with_voting(norm, reference_images, comparator, threshold=0.8)
        y_pred.append(result)
        y_true.append(1)

    # Проверка подделок
    for fname in test_forgeries:
        img = cv2.imread(os.path.join(forgeries_path, fname), 0)
        norm = SignatureNormalizer(img).normalize()
        result = verifier.verify_with_voting(norm, reference_images, comparator, threshold=0.8)
        y_pred.append(result)
        y_true.append(-1)

    # Метрики
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    tp = sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))
    fp = sum((np.array(y_pred) == 1) & (np.array(y_true) == -1))
    fn = sum((np.array(y_pred) == -1) & (np.array(y_true) == 1))
    tn = sum((np.array(y_pred) == -1) & (np.array(y_true) == -1))

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"📥 Recall: {rec:.4f}")
    print(f"📊 F1-score: {f1:.4f}")
    print(f"📌 TP={tp}, FP={fp}, FN={fn}, TN={tn}")

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
    all_metrics.append([acc, prec, rec, f1])

# Средние метрики
all_metrics = np.array(all_metrics)
avg_acc = np.mean(all_metrics[:, 0])
avg_prec = np.mean(all_metrics[:, 1])
avg_rec = np.mean(all_metrics[:, 2])
avg_f1 = np.mean(all_metrics[:, 3])

print("\n📊 Средние метрики по всем пользователям:")
print(f"✅ Средняя Accuracy:  {avg_acc:.4f}")
print(f"🎯 Средняя Precision: {avg_prec:.4f}")
print(f"📥 Средняя Recall:    {avg_rec:.4f}")
print(f"📊 Средняя F1-score:  {avg_f1:.4f}")
