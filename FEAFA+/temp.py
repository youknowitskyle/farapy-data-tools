import random

folds = {1: [], 2: [], 3: []}

options = [1, 2, 3]

for i in range(1, 128):
    folds[random.choice(options)].append(f"PV{i:03}")

print(folds)
