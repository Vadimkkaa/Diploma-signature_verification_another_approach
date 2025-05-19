import random

class SystemHealthMonitor:
    def __init__(self):
        self.cpu_load = 0
        self.ram_usage = 0

    def update_metrics(self):
        self.cpu_load = random.randint(5, 30)
        self.ram_usage = random.randint(300, 800)

    def check_overload(self):
        if self.cpu_load > 90 or self.ram_usage > 4000:
            print("Внимание: высокая нагрузка!")
        else:
            print("Система работает в нормальном режиме.")

    def log_status(self):
        print(f"CPU: {self.cpu_load}%, RAM: {self.ram_usage} МБ")

    def reset(self):
        self.cpu_load = 0
        self.ram_usage = 0
