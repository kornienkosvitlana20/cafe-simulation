import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# ================= БЛОК ПАРАМЕТРІВ (тримати в одному місці!) =================
STEP_MIN = 10                      # крок моделювання, хв
START_MIN = 8 * 60                 # відкриття о 08:00
LAM_MULT = 1.0                     # множник λ (для перевірки адекватності)

# (назва, кількість кроків по 10 хв, λ = клієнтів за 10 хв)
PERIODS = [("08-10", 12, 3.0),
           ("10-12", 12, 2.0),
           ("12-15", 18, 9.0),     # обід, пік
           ("15-18", 18, 3.0),
           ("18-20", 12, 4.0)]

# напій: (ціна у.о., час приготування хв, ймовірність)
MENU = {"еспресо":   (40, 2, 0.20),
        "американо": (45, 2, 0.25),
        "капучино":  (60, 4, 0.25),
        "лате":      (65, 5, 0.20),
        "чай":       (35, 3, 0.10)}
NAMES  = list(MENU)
PRICES = np.array([MENU[n][0] for n in NAMES])
TIMES  = np.array([MENU[n][1] for n in NAMES])
PROBS  = np.array([MENU[n][2] for n in NAMES])
assert abs(PROBS.sum() - 1) < 1e-9

P_GEOM = 0.6            # кількість напоїв ~ Geom(p), мін. 1, середнє 1/p
MAX_QUEUE = 10          # ліміт черги m
STATION_COST_H = 60     # вартість станції, у.о./год
MAX_STATIONS = 5
LOSS_LIMIT = 0.05       # умова для етапу 2; None = вимкнути
# ==============================================================================


class Client:
    def __init__(self, arrival_step):
        self.n_drinks = np.random.geometric(P_GEOM)
        self.drinks = np.random.choice(len(NAMES), size=self.n_drinks, p=PROBS)
        self.service_time = TIMES[self.drinks].sum()
        self.bill = PRICES[self.drinks].sum()
        self.arrival = arrival_step
        self.wait = 0.0


class Station:
    def __init__(self):
        self.client = None
        self.remaining = 0.0
        self.busy_time = 0.0

    def work(self, queue, step):
        done, left = [], STEP_MIN
        while left > 1e-9:
            if self.client is None:
                if not queue:
                    break
                self.client = queue.pop(0)
                self.remaining = self.client.service_time
                self.client.wait = (step - self.client.arrival) * STEP_MIN
            w = min(self.remaining, left)
            self.remaining -= w
            self.busy_time += w
            left -= w
            if self.remaining <= 1e-9:
                done.append(self.client)
                self.client = None
        return done


def run_day(active):
    """active — кількість активних станцій по періодах, напр. [1]*5 або [3,2,5,2,3]"""
    stations = [Station() for _ in range(MAX_STATIONS)]
    queue, rows, served, step = [], [], [], 0
    for p, (name, n_steps, lam) in enumerate(PERIODS):
        c = active[p]
        for _ in range(n_steps):
            arrived = np.random.poisson(lam * LAM_MULT)
            lost = lost_money = 0
            for _ in range(arrived):
                cl = Client(step)
                if len(queue) >= MAX_QUEUE:
                    lost += 1
                    lost_money += cl.bill
                else:
                    queue.append(cl)
            revenue = 0
            for i, st in enumerate(stations):
                if i < c or st.client is not None:
                    for cl in st.work(queue, step):
                        revenue += cl.bill
                        served.append(cl)
            t = START_MIN + step * STEP_MIN
            rows.append(dict(step=step, time=f"{t//60:02d}:{t%60:02d}", period=p,
                             active=c, arrived=arrived, queue=len(queue),
                             lost=lost, lost_money=lost_money, revenue=revenue,
                             cost=c * STEP_MIN / 60 * STATION_COST_H))
            step += 1
    return pd.DataFrame(rows), served


def run_many(active, n=100):
    res = []
    for _ in range(n):
        log, served = run_day(active)
        log["profit"] = log.revenue - log.cost
        r = dict(profit=log.profit.sum(),
                 loss_share=log.lost.sum() / max(log.arrived.sum(), 1),
                 avg_wait=np.mean([c.wait for c in served]) if served else 0,
                 avg_service=np.mean([c.service_time for c in served]) if served else 0,
                 avg_queue=log["queue"].mean())
        g = log.groupby("period")
        for p, v in g.profit.sum().items():
            r[f"p{p}"] = v
        for p, v in (g.lost.sum() / g.arrived.sum().clip(lower=1)).items():
            r[f"l{p}"] = v
        res.append(r)
    return pd.DataFrame(res)


# ------------------------------- ЕТАП 1 --------------------------------------
def stage1_single():
    np.random.seed(1)
    log, served = run_day([1] * 5)
    print(log.head(15).to_string())
    log.to_csv("figures/log_stage1.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(log.step, log["queue"])
    ax[0].set_title("Довжина черги за день")
    ax[0].set_xlabel("крок (10 хв)"); ax[0].set_ylabel("клієнтів у черзі")
    ax[1].hist([c.service_time for c in served], bins=15)
    ax[1].set_title("Час обслуговування, хв")
    ax[2].hist([c.n_drinks for c in served], bins=range(1, 9))
    ax[2].set_title("Кількість напоїв у замовленні")
    plt.tight_layout(); plt.savefig("figures/stage1_single.png", dpi=150)
    plt.close()


def stage1_many():
    r = run_many([1] * 5)
    cols = ["profit", "loss_share", "avg_wait", "avg_service", "avg_queue"]
    t = r[cols].agg(["mean", "std"]).round(3)
    print(t); t.to_csv("figures/stage1_summary.csv")
    r.loss_share.hist(bins=15)
    plt.title("Частка втрачених клієнтів, 100 прогонів")
    plt.xlabel("частка втрат"); plt.ylabel("кількість прогонів")
    plt.savefig("figures/stage1_loss_hist.png", dpi=150); plt.close()


def check(label, active=None, **over):
    """Один рядок таблиці адекватності: тимчасово змінює параметри."""
    active = active or [1] * 5
    old = {k: globals()[k] for k in over}
    globals().update(over)
    r = run_many(active, 100)
    globals().update(old)
    return dict(experiment=label, avg_queue=r.avg_queue.mean(),
                loss_share=r.loss_share.mean(), avg_wait=r.avg_wait.mean(),
                avg_service=r.avg_service.mean(), profit=r.profit.mean())


def stage1_check():
    rows = [check("базовий"),
            check("λ x2", LAM_MULT=2.0),
            check("MAX_QUEUE = 20", MAX_QUEUE=20),
            check("лише еспресо/американо", PROBS=np.array([.5, .5, 0, 0, 0])),
            check("λ x0.3 (спокійний день)", LAM_MULT=0.3),
            check("λ x0.15 (дуже спокійний день)", LAM_MULT=0.15)]
    t = pd.DataFrame(rows).round(3)
    print(t.to_string()); t.to_csv("figures/stage1_adequacy.csv", index=False)


# ------------------------------- ЕТАП 2 --------------------------------------
def stage2_check():
    rows = []
    for c in (1, 2, 3, 5):
        r = run_many([c] * 5)
        rows.append(dict(c=c, avg_queue=r.avg_queue.mean(), loss_share=r.loss_share.mean(),
                         cost=c * 12 * STATION_COST_H, profit=r.profit.mean()))
    t = pd.DataFrame(rows).round(3)
    print(t.to_string()); t.to_csv("figures/stage2_adequacy.csv", index=False)


def stage2():
    table = []
    prof = np.zeros((MAX_STATIONS, len(PERIODS)))
    loss = np.zeros((MAX_STATIONS, len(PERIODS)))
    for c in range(1, MAX_STATIONS + 1):
        r = run_many([c] * 5)
        table.append(dict(c=c, profit=r.profit.mean(), loss_share=r.loss_share.mean(),
                          avg_wait=r.avg_wait.mean(), avg_queue=r.avg_queue.mean()))
        prof[c - 1] = [r[f"p{i}"].mean() for i in range(len(PERIODS))]
        loss[c - 1] = [r[f"l{i}"].mean() for i in range(len(PERIODS))]
    t = pd.DataFrame(table).round(3)
    print(t.to_string()); t.to_csv("figures/stage2_table_c.csv", index=False)

    schedule = []
    for p in range(len(PERIODS)):
        ok = [c for c in range(MAX_STATIONS)
              if LOSS_LIMIT is None or loss[c, p] <= LOSS_LIMIT]
        cand = ok if ok else list(range(MAX_STATIONS))
        schedule.append(max(cand, key=lambda c: prof[c, p]) + 1)
    print("Розклад по періодах:", schedule)
    pd.DataFrame(dict(period=[p[0] for p in PERIODS], best_c=schedule,
                      profit_by_c=[list(prof[:, i].round(0)) for i in range(len(PERIODS))],
                      loss_by_c=[list(loss[:, i].round(3)) for i in range(len(PERIODS))])
                 ).to_csv("figures/stage2_schedule.csv", index=False)

    final = run_many(schedule)
    s = final[["profit", "loss_share", "avg_wait", "avg_queue"]].agg(["mean", "std"]).round(3)
    print(s); s.to_csv("figures/stage2_final.csv")

    t.plot(x="c", y="profit", kind="bar", legend=False)
    plt.ylabel("Середній прибуток за день, у.о."); plt.xlabel("кількість станцій c")
    plt.savefig("figures/stage2_profit.png", dpi=150); plt.close()

    np.random.seed(2)
    log, _ = run_day(schedule)
    print(log.head(15).to_string()); log.to_csv("figures/log_stage2.csv", index=False)
    plt.plot(log.step, log["queue"]); plt.title(f"Черга за день, розклад {schedule}")
    plt.xlabel("крок (10 хв)"); plt.ylabel("клієнтів у черзі")
    plt.savefig("figures/stage2_queue.png", dpi=150); plt.close()


if __name__ == "__main__":
    cmds = dict(single=stage1_single, many=stage1_many, check1=stage1_check,
                check2=stage2_check, stage2=stage2)
    name = sys.argv[1] if len(sys.argv) > 1 else "single"
    cmds[name]()