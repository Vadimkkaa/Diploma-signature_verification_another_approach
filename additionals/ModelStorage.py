import os

class ModelStorage:
    def __init__(self, base_path="storage/models"):
        self.base_path = base_path

    def save_model(self, user_id, model_obj):
        filename = f"user_{user_id}_model.pkl"
        print(f"Модель сохранена как {filename} в {self.base_path}")

    def load_model(self, user_id):
        filename = f"user_{user_id}_model.pkl"
        print(f"Модель загружена из {os.path.join(self.base_path, filename)}")
        return {}

    def delete_model(self, user_id):
        print(f"Модель user_{user_id} удалена.")

    def list_models(self):
        return [f"user_{i}_model.pkl" for i in range(1, 21)]
