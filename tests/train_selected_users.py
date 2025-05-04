
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.signature_normalizer import SignatureNormalizer
from core.signature_comparator import SignatureComparator
from core.ocs_verifier import OCSVMVerifier
from utils.results_logger import ResultsLogger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
originals_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "originals")
forgeries_path = os.path.join(BASE_DIR, "storage", "data", "CEDAR", "forgeries")

# выбор сразу нескольких для теста
selected_users = [5, 6, 8, 16]

comparator = SignatureComparator()
logger = ResultsLogger()

for user_id in selected_users:
    print(f"\n🔍 Тестируем пользователя {user_id}...")

    original_files = sorted([
        f for f in os.listdir(originals_path)
        if f.startswith(f"original_{user_id}_")
    ])

    forgery_files = sorted([
        f for f in os.listdir(forgeries_path)
        if f.startswith(f"forgeries_{user_id}_")
    ])

    if len(original_files) < 24 or len(forgery_files) < 24:
        print(f"⚠️ Недостаточно данных для пользователя {user_id}. Пропускаем.")
        continue

    train_files = original_files[:15]
    test_originals = original_files[15:]
    test_forgeries = forgery_files

    reference_images = []
    for fname in train_files:
        img = cv2.imread(os.path.join(originals_path, fname), 0)
        norm = SignatureNormalizer(img).normalize()
        reference_images.append(norm)

    pairwise_vectors = []
    for i in range(len(reference_images)):
        for j in range(i + 1, len(reference_images)):
            vec = comparator.compare(reference_images[i], reference_images[j])
            pairwise_vectors.append(vec)

    verifier = OCSVMVerifier()
    verifier.fit_on_pairs(pairwise_vectors)
    verifier.save_model(user_id)

    y_true, y_pred = [], []

    for fname in test_originals:
        img = cv2.imread(os.path.join(originals_path, fname), 0)
        norm = SignatureNormalizer(img).normalize()
        result = verifier.verify_with_voting(norm, reference_images, comparator)
        y_pred.append(result)
        y_true.append(1)

    for fname in test_forgeries:
        img = cv2.imread(os.path.join(forgeries_path, fname), 0)
        norm = SignatureNormalizer(img).normalize()
        result = verifier.verify_with_voting(norm, reference_images, comparator)
        y_pred.append(result)
        y_true.append(-1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"📥 Recall:    {rec:.4f}")
    print(f"📊 F1-score:  {f1:.4f}")
