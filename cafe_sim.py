"""
STEM-проєкт, варіант 7: кав'ярня / бариста.
Етап 1: n_stations = 1  (один канал)
Етап 2: n_stations = 1..5, розклад по періодах (багатоканальна система)
Покроковий модельний час, крок = 10 хв, 07:00-22:00.
"""
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- 1. ПАРАМЕТРИ (підставте свої, це приклад) ----------------
STEP = 10                                   # хв
START_H, END_H = 7, 22
N_STEPS = (END_H - START_H) * 60 // STEP    # 90 кроків

# (початок, кінець, lambda = сподівана кількість клієнтів за КРОК 10 хв)
PERIODS = [
    (7, 10, 5.0),    # ранковий пік
    (10, 12, 2.0),   # спад
    (12, 15, 7.0),   # обідній пік
    (15, 18, 3.0),   # денний спад
    (18, 22, 4.0),   # вечір
]

MENU = pd.DataFrame({
    "name":  ["Еспресо", "Американо", "Капучино", "Лате", "Чай"],
    "price": [45, 50, 70, 80, 40],          # грн
    "prep":  [2, 3, 4, 5, 3],               # хв приготування
    "prob":  [0.25, 0.20, 0.30, 0.15, 0.10] # сума = 1
})
P_GEOM = 0.6          # геометричний розподіл: к-сть напоїв у замовленні (>=1)
QUEUE_LIMIT = 15      # макс. довжина черги
STATION_COST_H = 150  # грн/год за одну станцію (разом із зарплатою бариста)


def period_of(hour):
    for i, (a, b, lam) in enumerate(PERIODS):
        if a <= hour < b:
            return i
    raise ValueError(hour)


# ---------------- 2. СУТНОСТІ ----------------
class Client:
    def __init__(self, arrival):
        self.arrival = arrival
        self.n_drinks = np.random.geometric(P_GEOM)
        idx = np.random.choice(len(MENU), size=self.n_drinks, p=MENU.prob)
        self.service = MENU.prep.values[idx].sum()    # хв
        self.price = MENU.price.values[idx].sum()     # грн


class Station:
    """Один канал обслуговування."""
    def __init__(self):
        self.left = 0.0        # скільки хв лишилось до завершення поточного замовлення
        self.busy = 0.0        # накопичений час зайнятості

    def work(self, queue, now, started):
        """Працює протягом одного кроку. started -- список клієнтів,
        яких станція взяла в роботу (для статистики)."""
        budget = STEP
        while budget > 0:
            if self.left <= 0:
                if not queue:
                    break                     # простій
                c = queue.popleft()
                self.left = c.service
                started.append((c, now + (STEP - budget) - c.arrival))
            used = min(budget, self.left)
            self.left -= used
            self.busy += used
            budget -= used


# ---------------- 3. ОДИН ПРОГІН ----------------
def run_day(stations_by_period):
    """stations_by_period -- список з к-стю активних станцій для кожного періоду,
    напр. [2, 1, 5, 2, 3]. Для етапу 1: [1, 1, 1, 1, 1]."""
    stations = [Station() for _ in range(max(stations_by_period))]
    queue = deque()
    rows = []
    for k in range(N_STEPS):
        now = k * STEP
        hour = START_H + now / 60
        p = period_of(hour)
        c_active = stations_by_period[p]

        # прибуття клієнтів
        arrived = np.random.poisson(PERIODS[p][2])
        lost = 0
        for _ in range(arrived):
            if len(queue) >= QUEUE_LIMIT:
                lost += 1
            else:
                queue.append(Client(now))

        # обслуговування активними станціями
        started = []
        for s in stations[:c_active]:
            s.work(queue, now, started)

        revenue = sum(c.price for c, _ in started)
        cost = c_active * STATION_COST_H * STEP / 60
        rows.append(dict(
            step=k, time=f"{int(hour):02d}:{int(round((hour % 1) * 60)):02d}",
            period=p, arrived=arrived, lost=lost, queue=len(queue),
            active=c_active, started=len(started), revenue=revenue, cost=cost,
            wait_sum=sum(w for _, w in started)))
    log = pd.DataFrame(rows)
    log["profit"] = log.revenue - log.cost
    return log


def day_metrics(log):
    served = log.started.sum()
    return dict(
        arrived=log.arrived.sum(), lost=log.lost.sum(), served=served,
        lost_share=log.lost.sum() / max(log.arrived.sum(), 1),
        revenue=log.revenue.sum(), cost=log.cost.sum(), profit=log.profit.sum(),
        avg_queue=log["queue"].mean(), max_queue=log["queue"].max(),
        avg_wait=log.wait_sum.sum() / max(served, 1))


# ---------------- 4. СЕРІЯ ПРОГОНІВ ----------------
def run_many(schedule, n_runs=100):
    logs = [run_day(schedule) for _ in range(n_runs)]
    metrics = pd.DataFrame([day_metrics(l) for l in logs])
    return logs, metrics


def mean_by_period(logs):
    """Середні (за прогонами) показники для кожного періоду."""
    all_logs = pd.concat(logs)
    g = all_logs.groupby("period")[["arrived", "lost", "revenue", "cost", "profit"]].sum()
    return g / len(logs)


# ---------------- 5. ЕТАП 1: ОДИН КАНАЛ ----------------
def stage1():
    np.random.seed(1)
    one = [1] * len(PERIODS)
    log = run_day(one)
    print(log.head(12))                        # фрагмент логу для звіту
    print(day_metrics(log))

    logs, m = run_many(one, 100)
    print("\nСередні за 100 прогонів (1 станція):")
    print(m.mean().round(2))
    print(m.std().round(2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(log.time, log["queue"]); ax[0].set_title("Довжина черги за день")
    ax[0].set_xticks(log.time[::10]); ax[0].set_xlabel("час")
    ax[1].hist(m.lost_share, bins=15); ax[1].set_title("Частка втрачених, 100 прогонів")
    plt.tight_layout(); plt.show()


# ---------------- 6. ЕТАП 2: БАГАТОКАНАЛЬНА ----------------
def stage2():
    np.random.seed(2)
    # 6.1. Однакова к-сть станцій на весь день: c = 1..5
    per_c = {}
    for c in range(1, 6):
        logs, m = run_many([c] * len(PERIODS), 100)
        per_c[c] = (logs, m)
        print(f"c={c}: прибуток={m.profit.mean():.0f}, "
              f"втрачено={m.lost_share.mean():.1%}, сер.очікування={m.avg_wait.mean():.1f} хв")

    # 6.2. Для кожного періоду обираємо c з найбільшим прибутком
    best = []
    for p in range(len(PERIODS)):
        profits = {c: mean_by_period(per_c[c][0]).loc[p, "profit"] for c in per_c}
        best.append(max(profits, key=profits.get))
    print("\nРозклад станцій по періодах:", dict(zip([f"{a}-{b}" for a, b, _ in PERIODS], best)))

    # 6.3. Перевірка розкладу на 100 прогонах
    logs, m = run_many(best, 100)
    print("\nОптимальний розклад, середнє за 100 прогонів:")
    print(m.mean().round(2))

    # Графік: прибуток від c
    cs = list(per_c)
    plt.bar(cs, [per_c[c][1].profit.mean() for c in cs])
    plt.xlabel("Кількість станцій"); plt.ylabel("грн/добу"); plt.title("Прибуток")
    plt.show()


if __name__ == "__main__":
    stage1()
    stage2()
