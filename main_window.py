import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog, QComboBox,
    QDateEdit, QTableWidget, QFormLayout, QStatusBar, QTabWidget
)
from PyQt5.QtGui import QPixmap
import os
from PyQt5.QtCore import Qt
from utils.user_registry import UserRegistry
from core.signature_normalizer import SignatureNormalizer
from core.signature_features import SignatureFeaturesExtractor
from core.ocs_verifier import OCSVMVerifier
from core.signature_comparator import SignatureComparator
from utils.results_logger import ResultsLogger
import numpy as np



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система проверки подписей")
        self.setGeometry(100, 100, 1000, 700)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.tabs = QTabWidget()
        self.init_tabs()
        self.setCentralWidget(self.tabs)
        self.load_btn.clicked.connect(self.load_image)
        self.user_registry = UserRegistry()
        self.populate_user_list()

        self.current_image_path = None

    def init_tabs(self):
        self.tab_verification = QWidget()
        self.tab_add_user = QWidget()
        self.tab_history = QWidget()
        self.tab_settings = QWidget()

        self.init_verification_tab()
        self.init_add_user_tab()
        self.init_history_tab()
        self.init_settings_tab()

        self.tabs.addTab(self.tab_verification, "Проверка подписи")
        self.tabs.addTab(self.tab_add_user, "Добавление пользователя")
        self.tabs.addTab(self.tab_history, "История проверок")
        self.tabs.addTab(self.tab_settings, "Настройки")

    def init_verification_tab(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Выберите изображение с подписью:"))
        self.load_btn = QPushButton("Загрузить файл")
        layout.addWidget(self.load_btn)

        self.preview_label = QLabel("Изображение не выбрано")
        self.preview_label.setFixedSize(300, 150)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        layout.addWidget(QLabel("Выберите пользователя:"))
        self.user_combo = QComboBox()
        layout.addWidget(self.user_combo)

        self.check_btn = QPushButton("Проверить подпись")
        self.check_btn.clicked.connect(self.verify_signature)  # ✅ подключаем после создания
        layout.addWidget(self.check_btn)

        layout.addWidget(QLabel("Результат:"))
        self.result_label = QLabel("-")
        layout.addWidget(self.result_label)

        layout.addWidget(QLabel("Технические метрики:"))
        self.metrics_text = QTextEdit()
        layout.addWidget(self.metrics_text)

        self.clear_btn = QPushButton("Очистить результат")
        self.clear_btn.clicked.connect(self.clear_result)
        layout.addWidget(self.clear_btn)

        self.tab_verification.setLayout(layout)

    def init_add_user_tab(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Введите данные пользователя:"))
        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("Фамилия")
        layout.addWidget(self.last_name_input)

        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("Имя")
        layout.addWidget(self.first_name_input)

        self.middle_name_input = QLineEdit()
        self.middle_name_input.setPlaceholderText("Отчество")
        layout.addWidget(self.middle_name_input)

        self.birth_date_input = QDateEdit()
        self.birth_date_input.setCalendarPopup(True)
        layout.addWidget(self.birth_date_input)

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["мужской", "женский"])
        layout.addWidget(self.gender_combo)

        layout.addWidget(QLabel("Загрузите эталонные подписи:"))
        self.train_btn = QPushButton("Выбрать папку")
        layout.addWidget(self.train_btn)

        self.train_confirm_btn = QPushButton("Обучить модель")
        layout.addWidget(self.train_confirm_btn)

        self.train_status = QTextEdit()
        layout.addWidget(self.train_status)

        self.tab_add_user.setLayout(layout)

    def init_history_tab(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Фильтр по пользователю:"))
        self.history_user_combo = QComboBox()
        layout.addWidget(self.history_user_combo)

        self.history_table = QTableWidget()
        layout.addWidget(self.history_table)

        self.refresh_history_btn = QPushButton("Обновить таблицу")
        layout.addWidget(self.refresh_history_btn)

        self.tab_history.setLayout(layout)

    def init_settings_tab(self):
        layout = QFormLayout()

        self.model_path_input = QLineEdit()
        self.db_path_input = QLineEdit()
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Русский", "English"])

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Светлая", "Тёмная"])

        layout.addRow("Путь к моделям:", self.model_path_input)
        layout.addRow("Путь к базе данных:", self.db_path_input)
        layout.addRow("Язык интерфейса:", self.language_combo)
        layout.addRow("Тема оформления:", self.theme_combo)

        self.save_settings_btn = QPushButton("Сохранить настройки")
        layout.addRow(self.save_settings_btn)

        self.tab_settings.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Изображения (*.png *.jpg *.jpeg)"
        )
        if file_path:
            pixmap = QPixmap(file_path)
            scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
            self.preview_label.setText("")  # очищаем текст
            self.statusBar.showMessage(f"Файл загружен: {os.path.basename(file_path)}")
            self.current_image_path = file_path
        else:
            self.statusBar.showMessage("Файл не выбран")

    def populate_user_list(self):
        self.user_combo.clear()
        users = self.user_registry.get_all_users()  # возвращает список словарей
        self.user_map = {}  # отображение ФИО → user_id

        for user in users:
            full_name = f"{user['last_name']} {user['first_name']} {user['middle_name']}"
            self.user_combo.addItem(full_name)
            self.user_map[full_name] = user["user_id"]

    def verify_signature(self):
        from core.signature_loader import SignatureLoader
        from core.signature_normalizer import SignatureNormalizer
        from core.signature_features import SignatureFeaturesExtractor
        from core.signature_comparator import SignatureComparator
        from core.ocs_verifier import OCSVMVerifier
        import os
        import cv2

        # Проверка: выбран пользователь?
        selected_name = self.user_combo.currentText()
        user_id = self.user_map.get(selected_name)
        if not user_id:
            self.statusBar.showMessage("❌ Не выбран пользователь")
            return

        # Проверка: загружен файл?
        if not hasattr(self, "current_image_path") or not self.current_image_path:
            self.statusBar.showMessage("❌ Изображение не загружено")
            return
        print("🔎 user_id:", user_id)
        print("📂 Проверяется файл:", self.current_image_path)

        try:
            # 1. Загрузка изображения
            loader = SignatureLoader(self.current_image_path)
            image = loader.load_image()

            # 2. Нормализация
            normalizer = SignatureNormalizer(image)
            normalized = normalizer.normalize()

            # 3. Извлечение признаков
            extractor = SignatureFeaturesExtractor(normalized)

            lbp = extractor.extract_lbp_features()
            curv = extractor.extract_curvature_features()
            hog = extractor.extract_hog_features()

            features = np.concatenate([lbp, curv, hog])

            # 4. Сравнение с эталонами
            verifier = OCSVMVerifier()
            try:
                verifier.load_model(user_id)
            except FileNotFoundError:
                self.statusBar.showMessage("❌ Модель не найдена")
                return


            comparator = SignatureComparator()


            original_dir = os.path.join("storage", "data", "CEDAR", "originals")
            reference_images = []

            for i in range(1, 16):  # 15 эталонов
                file = f"original_{user_id}_{i}.png"
                path = os.path.join(original_dir, file)
                if os.path.exists(path):
                    ref_img = cv2.imread(path, 0)
                    if ref_img is not None:
                        norm_ref = SignatureNormalizer(ref_img).normalize()
                        reference_images.append(norm_ref)

            if not reference_images:
                self.statusBar.showMessage("❌ Эталонные подписи не найдены")
                return


            result = verifier.verify_with_voting(
                new_signature_image=normalized,
                reference_images=reference_images,
                comparator=comparator
            )

            # 5. Вывод результата
            if result == 1:
                self.result_label.setText("✅ Подпись ПРИНЯТА")
            else:
                self.result_label.setText("❌ Подпись ОТКЛОНЕНА")

            self.statusBar.showMessage("✅ Проверка завершена")


        except Exception as e:
            self.statusBar.showMessage(f"❌ Ошибка: {str(e)}")

    def clear_result(self):
        self.result_label.setText("-")
        self.metrics_text.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
