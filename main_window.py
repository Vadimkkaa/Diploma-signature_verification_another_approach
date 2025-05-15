import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog, QComboBox,
    QDateEdit, QTableWidget, QFormLayout, QStatusBar, QTabWidget,
    QTableWidgetItem
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
from core.signature_loader import SignatureLoader

from utils.config_manager import ConfigManager


import numpy as np
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система проверки подписей")
        self.setGeometry(100, 100, 1000, 700)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.translations = {
            "Русский": {
                "tabs": ["Проверка подписи", "Пользователи", "История проверок", "Настройки"],
                "verify_title": "Выберите изображение с подписью:",
                "verify_button": "Загрузить файл",
                "preview_placeholder": "Изображение не выбрано",
                "select_user": "Выберите пользователя:",
                "check_button": "Проверить подпись",
                "result_label": "Результат:",
                "metrics_label": "Технические метрики:",
                "clear_button": "Очистить результат",
                "save_status": "✅ Настройки сохранены",
                "Accepted": "✅ Подпись ПРИНЯТА",
                "Rejected": "❌ Подпись ОТКЛОНЕНА",
                "Finished": "✅ Проверка завершена",
                "tab_user": "Пользователи",
                "user_input": "Введите данные пользователя:",
                "last_name": "Фамилия",
                "first_name": "Имя",
                "middle_name": "Отчество",
                "gender_male": "мужской",
                "gender_female": "женский",
                "upload_signatures": "Загрузите эталонные подписи:",
                "select_folder": "Выбрать папку",
                "train_model": "Обучить модель",
                "delete_user": "Удаление пользователя:",
                "delete_btn": "Удалить пользователя",
                "tab_history": "История проверок",
                "filter_by_user": "Фильтр по пользователю:",
                "filter_by_id": "Фильтр по ID пользователя:",
                "filter_id_placeholder": "Введите user_id",
                "apply_filter": "Применить фильтр по ID",
                "refresh_table": "Обновить таблицу",
                "tab_settings": "Настройки",
                "interface_language": "Язык интерфейса:",
                "theme": "Тема оформления:",
                "save_settings": "Сохранить настройки",
                "theme_light": "Светлая",
                "theme_dark": "Тёмная",
                "filter_user": "Фильтр по пользователю:",
                "filter_id": "Фильтр по ID пользователя:",
                "all_users": "Все пользователи",
                "headers": ["ID", "Пользователь", "Результат", "Голоса 'за'", "Порог", "Дата/время"],
                "accepted": "✅ Принята",
                "rejected": "❌ Отклонена",
                "id_placeholder": "Введите user_id",
                "apply_id_filter": "Применить фильтр по ID",
                "refresh_btn": "Обновить таблицу"

            },
            "English": {
                "tabs": ["Signature Verification", "Users", "Verification History", "Settings"],
                "verify_title": "Select a signature image:",
                "verify_button": "Load File",
                "preview_placeholder": "No image selected",
                "select_user": "Select user:",
                "check_button": "Verify Signature",
                "result_label": "Result:",
                "metrics_label": "Technical Metrics:",
                "clear_button": "Clear Result",
                "save_status": "✅ Settings saved",
                "Accepted": "✅ Signature ACCEPTED",
                "Rejected": "❌ Signature REJECTED",
                "Finished": "✅ Verification completed",
                "tab_user": "Users",
                "user_input": "Enter user data:",
                "last_name": "Last Name",
                "first_name": "First Name",
                "middle_name": "Middle Name",
                "gender_male": "male",
                "gender_female": "female",
                "upload_signatures": "Upload reference signatures:",
                "select_folder": "Select Folder",
                "train_model": "Train Model",
                "delete_user": "Delete user:",
                "delete_btn": "Delete User",
                "tab_history": "Verification History",
                "filter_by_user": "Filter by user:",
                "filter_by_id": "Filter by user ID:",
                "filter_id_placeholder": "Enter user_id",
                "apply_filter": "Apply ID filter",
                "refresh_table": "Refresh table",
                "tab_settings": "Settings",
                "interface_language": "Interface language:",
                "theme": "Theme:",
                "save_settings": "Save settings",
                "theme_light": "Light",
                "theme_dark": "Dark",
                "filter_user": "Filter by user:",
                "filter_id": "Filter by user ID:",
                "all_users": "All users",
                "headers": ["ID", "User", "Result", "Votes 'for'", "Threshold", "Date/Time"],
                "accepted": "✅ Accepted",
                "rejected": "❌ Rejected",
                "id_placeholder": "Enter user_id",
                "apply_id_filter": "Apply ID filter",
                "refresh_btn": "Refresh table"

            }
        }

        self.tabs = QTabWidget()
        self.init_tabs()
        self.setCentralWidget(self.tabs)
        self.load_btn.clicked.connect(self.load_image)
        self.user_registry = UserRegistry()
        self.populate_user_list()
        self.refresh_history_btn.clicked.connect(self.load_verification_history)
        self.history_user_combo.currentIndexChanged.connect(self.load_verification_history)
        self.load_verification_history()

        self.update_delete_user_combo()

        self.current_image_path = None

        self.train_btn.clicked.connect(self.select_training_folder)
        self.train_folder = None

        self.train_confirm_btn.clicked.connect(self.train_model)

        self.config = ConfigManager()
        settings = self.config.load_config()
        self.language_combo.setCurrentText(settings["language"])
        self.theme_combo.setCurrentText(settings["theme"])
        self.apply_theme()



        self.result_text_label = QLabel()
        self.metrics_label = QLabel()

        self.apply_language()

    def init_tabs(self):
        self.tab_verification = QWidget()
        self.tab_user = QWidget()
        self.tab_history = QWidget()
        self.tab_settings = QWidget()

        self.init_verification_tab()
        self.init_user_tab()
        self.init_history_tab()
        self.init_settings_tab()

        self.tabs.addTab(self.tab_verification, "Проверка подписи")
        self.tabs.addTab(self.tab_user, "Пользователи")
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

        self.result_text_label = QLabel()  # ← просто пустой
        layout.addWidget(self.result_text_label)

        self.result_label = QLabel("-")
        layout.addWidget(self.result_label)

        self.metrics_label = QLabel()
        layout.addWidget(self.metrics_label)

        self.metrics_text = QTextEdit()
        layout.addWidget(self.metrics_text)

        self.clear_btn = QPushButton("Очистить результат")
        self.clear_btn.clicked.connect(self.clear_result)
        layout.addWidget(self.clear_btn)

        self.tab_verification.setLayout(layout)

    def init_user_tab(self):
        layout = QVBoxLayout()

        # Заголовок: Введите данные пользователя
        self.user_input_label = QLabel("Введите данные пользователя:")
        layout.addWidget(self.user_input_label)

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

        # Загрузка подписей
        self.signature_folder_label = QLabel("Загрузите эталонные подписи:")
        layout.addWidget(self.signature_folder_label)

        self.train_btn = QPushButton("Выбрать папку")
        layout.addWidget(self.train_btn)

        self.train_confirm_btn = QPushButton("Обучить модель")
        layout.addWidget(self.train_confirm_btn)

        self.train_status = QTextEdit()
        layout.addWidget(self.train_status)

        # Удаление пользователя
        self.delete_user_label = QLabel("Удаление пользователя:")
        layout.addWidget(self.delete_user_label)

        self.delete_user_combo = QComboBox()
        layout.addWidget(self.delete_user_combo)

        self.delete_user_btn = QPushButton("Удалить пользователя")
        self.delete_user_btn.clicked.connect(self.delete_user)
        layout.addWidget(self.delete_user_btn)

        self.tab_user.setLayout(layout)

    def init_history_tab(self):
        layout = QVBoxLayout()

        self.history_user_label = QLabel("Фильтр по пользователю:")
        layout.addWidget(self.history_user_label)

        self.history_user_combo = QComboBox()
        layout.addWidget(self.history_user_combo)

        self.user_id_label = QLabel("Фильтр по ID пользователя:")
        layout.addWidget(self.user_id_label)

        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("Введите user_id")
        layout.addWidget(self.user_id_input)

        self.apply_id_filter_btn = QPushButton("Применить фильтр по ID")
        layout.addWidget(self.apply_id_filter_btn)

        self.history_table = QTableWidget()
        layout.addWidget(self.history_table)

        self.refresh_history_btn = QPushButton("Обновить таблицу")
        layout.addWidget(self.refresh_history_btn)

        self.tab_history.setLayout(layout)

        self.apply_id_filter_btn.clicked.connect(self.load_verification_by_id)

    def init_settings_tab(self):
        layout = QFormLayout()

        self.language_combo = QComboBox()
        self.language_combo.addItems(["Русский", "English"])

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Светлая", "light")  # userData = "light"
        self.theme_combo.addItem("Тёмная", "dark")  # userData = "dark"

        self.language_label = QLabel("Язык интерфейса:")
        self.theme_label = QLabel("Тема оформления:")

        layout.addRow(self.language_label, self.language_combo)
        layout.addRow(self.theme_label, self.theme_combo)

        self.save_settings_btn = QPushButton("Сохранить настройки")
        layout.addRow(self.save_settings_btn)

        self.tab_settings.setLayout(layout)
        self.save_settings_btn.clicked.connect(self.save_ui_config)

    def save_ui_config(self):
        language = self.language_combo.currentText()
        theme = self.theme_combo.currentText()
        self.config.save_config(language, theme)
        self.apply_theme()
        self.apply_language()
        t = self.translations.get(language, self.translations["Русский"])
        self.statusBar.showMessage(t["save_status"])

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
        users = self.user_registry.get_all_users()
        self.user_map = {}

        # Сортируем по фамилии
        for user in sorted(users, key=lambda u: u['last_name']):
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

            result, metrics = verifier.verify_with_voting(
                new_signature_image=normalized,
                reference_images=reference_images,
                comparator=comparator,
                return_metrics=True
            )

            # 5. Вывод результата
            if result == 1:
                self.result_label.setText("✅ Подпись ПРИНЯТА")

            else:
                self.result_label.setText("❌ Подпись ОТКЛОНЕНА")

            percent = metrics["votes_for"] / metrics["total"] * 100
            text = f"""Проголосовало "За": {metrics['votes_for']} из {metrics['total']} = {percent:.2f}%
            Порог принятия: ≥ {int(metrics['threshold'] * 100)}%"""
            self.metrics_text.setText(text)

            self.statusBar.showMessage("✅ Проверка завершена")

            # Логирование в verification_logs
            logger = ResultsLogger()
            logger.log_verification(
                user_id=user_id,
                result=result,
                votes_for=metrics["votes_for"],
                threshold=metrics["threshold"]
            )



        except Exception as e:
            self.statusBar.showMessage(f"❌ Ошибка: {str(e)}")

    def clear_result(self):
        self.result_label.setText("-")
        self.metrics_text.clear()

    def select_training_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с подписями")
        if folder:
            self.train_folder = folder
            self.statusBar.showMessage(f"Папка выбрана: {folder}")
            self.train_status.append(f"✅ Выбрана папка: {folder}")
        else:
            self.statusBar.showMessage("Папка не выбрана")
            self.train_status.append("⚠️ Папка не выбрана")

    def train_model(self):
        from datetime import datetime

        # Проверка папки
        if not self.train_folder:
            self.statusBar.showMessage("⚠️ Папка с подписями не выбрана")
            self.train_status.append("❌ Сначала выберите папку с подписями.")
            return

        # Сбор данных пользователя
        last_name = self.last_name_input.text().strip()
        first_name = self.first_name_input.text().strip()
        middle_name = self.middle_name_input.text().strip()
        birth_date = self.birth_date_input.date().toString("yyyy-MM-dd")
        gender = self.gender_combo.currentText()

        if not all([last_name, first_name, middle_name]):
            self.statusBar.showMessage("❌ Заполните все поля ФИО")
            self.train_status.append("❌ Не все поля ФИО заполнены.")
            return

        # Генерация нового user_id
        all_users = self.user_registry.list_users()

        #Добавление в БД
        user_id = self.user_registry.add_user(
            last_name=last_name,
            first_name=first_name,
            middle_name=middle_name,
            gender=gender,
            birth_date=birth_date,
            return_id=True
        )

        self.statusBar.showMessage(f"✅ Пользователь {user_id} добавлен")
        self.train_status.append(f"✅ Добавлен пользователь ID={user_id}: {last_name} {first_name} {middle_name}")

        # Сохраняем user_id на будущее (для обучения модели)
        self.current_training_user_id = user_id

        # 1. Чтение и фильтрация изображений
        files = sorted([
            os.path.join(self.train_folder, f)
            for f in os.listdir(self.train_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if len(files) < 15:
            self.train_status.append("❌ Недостаточно подписей для обучения (нужно ≥ 15)")
            return

        # 2. Оставляем первые 15 файлов для обучения
        selected_files = files[:15]

        # 3. Нормализация изображений
        images = []
        for path in selected_files:
            try:
                img = cv2.imread(path, 0)
                if img is not None:
                    norm = SignatureNormalizer(img).normalize()
                    images.append(norm)
                else:
                    self.train_status.append(f"⚠️ Не удалось загрузить: {os.path.basename(path)}")
            except Exception as e:
                self.train_status.append(f"⚠️ Ошибка при {os.path.basename(path)}: {str(e)}")

        self.train_status.append(f"✅ Обработано и нормализовано {len(images)} изображений")

        # 4. Построение векторов сравнения (все пары i < j)
        comparator = SignatureComparator()
        pairs = []

        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                vec = comparator.compare(images[i], images[j])
                pairs.append(vec)

        self.train_status.append(f"🔧 Построено {len(pairs)} векторов признаков сходства")

        # 5. Обучение и сохранение модели
        verifier = OCSVMVerifier()
        verifier.fit_on_pairs(pairs)
        verifier.save_model(self.current_training_user_id)

        self.train_status.append(f"✅ Модель сохранена как user_{self.current_training_user_id}_model.pkl")
        self.statusBar.showMessage(f"✅ Обучение завершено для user_id={self.current_training_user_id}")
        self.populate_user_list()
        self.update_delete_user_combo()

    def delete_user(self):
        selected_name = self.delete_user_combo.currentText()
        if not selected_name:
            self.statusBar.showMessage("❌ Пользователь не выбран для удаления")
            return

        # Ищем user_id
        users = self.user_registry.get_all_users()
        user_map = {
            f"{u['last_name']} {u['first_name']} {u['middle_name']}": u['user_id']
            for u in users
        }
        user_id = user_map.get(selected_name)

        if not user_id:
            self.statusBar.showMessage("❌ Ошибка: пользователь не найден")
            return

        # Удаление из БД
        self.user_registry.delete_user(user_id)

        # Удаление модели
        model_path = os.path.join("storage", "models", f"user_{user_id}_model.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)

        self.statusBar.showMessage(f"✅ Пользователь {selected_name} удалён")
        self.train_status.append(f"🗑️ Удалён пользователь {selected_name} (ID={user_id})")

        # Обновляем все списки
        self.populate_user_list()
        self.update_delete_user_combo()

    def update_delete_user_combo(self):
        self.delete_user_combo.clear()
        users = self.user_registry.get_all_users()

        # сортируем по фамилии
        for u in sorted(users, key=lambda x: x['last_name']):
            full_name = f"{u['last_name']} {u['first_name']} {u['middle_name']}"
            self.delete_user_combo.addItem(full_name)

    def load_verification_history(self):
        logger = ResultsLogger()
        records = logger.get_verification_logs()

        # Получаем язык и переводы
        lang = self.language_combo.currentText()
        t = self.translations.get(lang, self.translations["Русский"])

        # Получаем пользователей
        users = self.user_registry.get_all_users()
        user_map = {u["user_id"]: f"{u['last_name']} {u['first_name']} {u['middle_name']}" for u in users}
        name_to_id = {v: k for k, v in user_map.items()}

        # 🔁 Обновляем список каждый раз (включая строку "Все пользователи")
        current_selected = self.history_user_combo.currentText()
        self.history_user_combo.blockSignals(True)
        self.history_user_combo.clear()
        self.history_user_combo.addItem(t["all_users"])
        for name in sorted(name_to_id.keys()):
            self.history_user_combo.addItem(name)
        self.history_user_combo.setCurrentText(current_selected)
        self.history_user_combo.blockSignals(False)

        # 🔍 Фильтрация по выбранному пользователю
        selected_name = self.history_user_combo.currentText()
        if selected_name != t["all_users"]:
            selected_id = name_to_id.get(selected_name)
            records = [r for r in records if r[1] == selected_id]

        # Очистка и настройка таблицы
        self.history_table.clearContents()
        self.history_table.setRowCount(0)
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(t["headers"])
        self.history_table.setRowCount(len(records))

        for row_idx, record in enumerate(records):
            log_id, uid, result, votes_for, threshold, timestamp = record
            fio = user_map.get(uid, f"user_{uid}")
            verdict = t["accepted"] if result == 1 else t["rejected"]

            self.history_table.setItem(row_idx, 0, QTableWidgetItem(str(uid)))
            self.history_table.setItem(row_idx, 1, QTableWidgetItem(fio))
            self.history_table.setItem(row_idx, 2, QTableWidgetItem(verdict))
            self.history_table.setItem(row_idx, 3, QTableWidgetItem(str(votes_for)))
            self.history_table.setItem(row_idx, 4, QTableWidgetItem(f"{threshold:.2f}"))
            self.history_table.setItem(row_idx, 5, QTableWidgetItem(timestamp))

    def load_verification_by_id(self):
        from utils.results_logger import ResultsLogger

        input_text = self.user_id_input.text().strip()
        if not input_text.isdigit():
            self.statusBar.showMessage("❌ Введите корректный числовой user_id")
            return

        user_id = int(input_text)
        logger = ResultsLogger()
        records = logger.get_verification_logs()
        filtered = [r for r in records if r[1] == user_id]

        # Получаем ФИО (если есть)
        users = self.user_registry.get_all_users()
        user_map = {u["user_id"]: f"{u['last_name']} {u['first_name']} {u['middle_name']}" for u in users}

        self.history_table.clearContents()
        self.history_table.setRowCount(0)
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(
            ["ID", "Пользователь", "Результат", "Голоса 'за'", "Порог", "Дата/время"])
        self.history_table.setRowCount(len(filtered))

        for row_idx, record in enumerate(filtered):
            log_id, uid, result, votes_for, threshold, timestamp = record
            fio = user_map.get(uid, f"user_{uid}")
            verdict = "✅ Принята" if result == 1 else "❌ Отклонена"

            self.history_table.setItem(row_idx, 0, QTableWidgetItem(str(uid)))
            self.history_table.setItem(row_idx, 1, QTableWidgetItem(fio))
            self.history_table.setItem(row_idx, 2, QTableWidgetItem(verdict))
            self.history_table.setItem(row_idx, 3, QTableWidgetItem(str(votes_for)))
            self.history_table.setItem(row_idx, 4, QTableWidgetItem(f"{threshold:.2f}"))
            self.history_table.setItem(row_idx, 5, QTableWidgetItem(timestamp))

    def apply_theme(self):
        theme = self.theme_combo.itemData(self.theme_combo.currentIndex())
        if theme == "dark":
            self.setStyleSheet("""
                   QWidget {
                       background-color: #2e2e2e;
                       color: white;
                   }
                   QLineEdit, QTextEdit, QComboBox, QDateEdit {
                       background-color: #3c3c3c;
                       color: white;
                       border: 1px solid #555;
                   }
                   QPushButton {
                       background-color: #444;
                       color: white;
                       border: 1px solid #666;
                   }
                   QTabWidget::pane {
                       border: 1px solid #666;
                   }
                   QTabBar::tab {
                       background: #3c3c3c;
                       color: white;
                       border: 1px solid #555;
                       padding: 5px;
                   }
                   QTabBar::tab:selected {
                       background: #5c5c5c;
                       border-bottom: 2px solid #00bcd4;
                   }
                   QHeaderView::section {
                       background-color: #444;
                       color: white;
                       padding: 4px;
                       border: 1px solid #666;
                   }
                   QTableWidget {
                       gridline-color: #666;
                       background-color: #2e2e2e;
                       color: white;
                       selection-background-color: #555;
                       selection-color: white;
                   }
                   QHeaderView::section {
                       background-color: #444;
                       color: white;
                       border: 1px solid #666;
                   }
                   QTableCornerButton::section {
                       background-color: #444;
                       border: 1px solid #666;
                   }
               """)
        else:
            self.setStyleSheet("")

    def apply_language(self):
        lang = self.language_combo.currentText()
        t = self.translations.get(lang, self.translations["Русский"])

        self.tabs.setTabText(0, t["tabs"][0])
        self.tabs.setTabText(1, t["tabs"][1])
        self.tabs.setTabText(2, t["tabs"][2])
        self.tabs.setTabText(3, t["tabs"][3])

        self.load_btn.setText(t["verify_button"])
        self.check_btn.setText(t["check_button"])
        self.clear_btn.setText(t["clear_button"])

        self.result_label.setText("-")
        self.metrics_text.setPlaceholderText(t["metrics_label"])

        self.statusBar.showMessage(t["save_status"])

        self.tab_verification.layout().itemAt(0).widget().setText(t["verify_title"])
        self.tab_verification.layout().itemAt(3).widget().setText(t["select_user"])
        self.tab_verification.layout().itemAt(7).widget().setText(t["result_label"])
        self.tab_verification.layout().itemAt(9).widget().setText(t["metrics_label"])

        self.preview_label.setText(t["preview_placeholder"])

        self.result_text_label.setText(t["result_label"])
        self.metrics_label.setText(t["metrics_label"])
        self.theme_combo.setItemText(0, t.get("theme_light", "Светлая"))
        self.theme_combo.setItemText(1, t.get("theme_dark", "Тёмная"))
        self.apply_language_user_tab(t)
        self.apply_language_history_tab(t)
        self.apply_language_settings_tab(t)
        self.statusBar.clearMessage()
        self.load_verification_history()

    def apply_language_user_tab(self, t):
        self.tabs.setTabText(1, t["tab_user"])  # Название вкладки

        self.user_input_label.setText(t["user_input"])
        self.last_name_input.setPlaceholderText(t["last_name"])
        self.first_name_input.setPlaceholderText(t["first_name"])
        self.middle_name_input.setPlaceholderText(t["middle_name"])
        self.birth_date_input.setDisplayFormat("dd/MM/yyyy")
        self.gender_combo.setItemText(0, t["gender_male"])
        self.gender_combo.setItemText(1, t["gender_female"])

        self.signature_folder_label.setText(t["upload_signatures"])
        self.train_btn.setText(t["select_folder"])
        self.train_confirm_btn.setText(t["train_model"])
        self.delete_user_label.setText(t["delete_user"])
        self.delete_user_btn.setText(t["delete_btn"])

    def apply_language_history_tab(self, t):
        self.tabs.setTabText(2, t["tabs"][2])  # Название вкладки

        self.history_user_combo.blockSignals(True)
        self.history_user_combo.setItemText(0, t["all_users"])  # Обновляем "Все пользователи"
        self.history_user_combo.blockSignals(False)

        self.tab_history.layout().itemAt(0).widget().setText(t["filter_user"])
        self.tab_history.layout().itemAt(2).widget().setText(t["filter_id"])
        self.user_id_input.setPlaceholderText(t["id_placeholder"])
        self.apply_id_filter_btn.setText(t["apply_id_filter"])
        self.refresh_history_btn.setText(t["refresh_btn"])

        # Если уже отображена таблица — обновить заголовки:
        self.history_table.setHorizontalHeaderLabels(t["headers"])

    def apply_language_settings_tab(self, t):
        self.tabs.setTabText(3, t["tab_settings"])
        self.language_label.setText(t["interface_language"])
        self.theme_label.setText(t["theme"])
        self.save_settings_btn.setText(t["save_settings"])
        self.theme_combo.setItemText(0, t["theme_light"])
        self.theme_combo.setItemText(1, t["theme_dark"])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
