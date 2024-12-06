import csv
import numpy as np

VUS_PRs_against_HP: dict[str, list[float]] = {}

with open("eval/HP_tuning/uni/MDRS.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        HP = row["HP"]
        VUS_PR = float(row["VUS-PR"])
        
        VUS_PRs = VUS_PRs_against_HP.get(HP, [])
        VUS_PRs.append(VUS_PR)
        VUS_PRs_against_HP[HP] = VUS_PRs

avgs: dict[str, float] = {}
for key, value in VUS_PRs_against_HP.items():
    value = np.array(value)
    avgs[key] = np.mean(value)

sorted_avgs = sorted(avgs.items(), key=lambda avg: avg[1])
print(sorted_avgs)
