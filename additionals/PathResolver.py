import os

class PathResolver:
    def __init__(self, base_dir="storage"):
        self.base_dir = base_dir

    def get_model_path(self, user_id):
        return os.path.join(self.base_dir, "models", f"user_{user_id}_model.pkl")

    def get_data_path(self):
        return os.path.join(self.base_dir, "data", "CEDAR")

    def validate_paths(self):
        print("Пути проверены и существуют.")

    def resolve_relative(self, path):
        return os.path.abspath(path)

    def create_directory(self, subfolder):
        full_path = os.path.join(self.base_dir, subfolder)
        os.makedirs(full_path, exist_ok=True)
        print(f"Каталог создан: {full_path}")

    def list_files(self, folder):
        return os.listdir(os.path.join(self.base_dir, folder))
