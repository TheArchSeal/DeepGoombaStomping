import os
import matplotlib.pyplot as plt

DATA_DIR = ".\\plots"

for file in os.listdir(DATA_DIR):
    with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
        points = [line.split() for line in f.readlines()]

        points_x = [int(p[0]) for p in points]
        points_y = [float(p[1]) for p in points]

        plt.plot(points_x, points_y, label=file.removesuffix(".txt"))

plt.xlabel("number of iterations")
plt.ylabel("total reward")
plt.legend()
plt.show()
