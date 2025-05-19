class AppConfig:
    def __init__(self):
        self.settings = {
            "language": "Русский",
            "theme": "Светлая"
        }

    def load(self):
        print("Настройки загружены из файла.")
        return self.settings

    def save(self, new_settings):
        self.settings.update(new_settings)
        print(f"Настройки сохранены: {self.settings}")

    def get(self, key):
        return self.settings.get(key)

    def set(self, key, value):
        self.settings[key] = value

    def reset_defaults(self):
        self.settings = {"language": "Русский", "theme": "Светлая"}
        print("Настройки сброшены по умолчанию.")
