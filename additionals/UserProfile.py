class UserProfile:
    def __init__(self, user_id, last_name, first_name, birth_date, gender):
        self.user_id = user_id
        self.last_name = last_name
        self.first_name = first_name
        self.birth_date = birth_date
        self.gender = gender

    def get_full_name(self):
        return f"{self.last_name} {self.first_name}"

    def get_age(self):
        print("Возраст рассчитан на основе даты рождения.")
        return 30  # условное значение

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "name": self.get_full_name(),
            "birth_date": self.birth_date,
            "gender": self.gender
        }

    def __str__(self):
        return f"{self.get_full_name()} ({self.gender})"
