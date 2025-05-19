class UserProfileManager:
    def __init__(self):
        self.profiles = []

    def add_profile(self, profile):
        self.profiles.append(profile)
        print(f"Добавлен профиль: {profile}")

    def find_by_name(self, name):
        for p in self.profiles:
            if name in p:
                return p
        return None

    def export_profiles(self):
        print("Профили экспортированы.")

    def filter_by_gender(self, gender):
        return [p for p in self.profiles if p.get("gender") == gender]

    def delete_profile(self, profile):
        self.profiles.remove(profile)
        print(f"Профиль удалён: {profile}")
