import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from operator import add
from functools import reduce
import time

# Options
images = [
    "e_t1.png",
    "e_t2.png",
    "e_t3.png",
    "e_t4.png",
]
entities = [
    {'label' : 'e1', 'color' : (200, 147, 199)},
    {'label' : 'e2', 'color' : (203, 100, 100)},
    {'label' : 'e3', 'color' : (229, 213, 140)}
]
N = 30
rewardOutFile = "reward.txt"

# Convert image to 2D grid of entities
emaps = [None for i in images]
for i in range(len(images)):
    img = Image.open(images[i])
    data = np.asarray(img)[:,:,:3]
    emap = np.zeros((data.shape[0], data.shape[1]))

    for r in range(emap.shape[0]):
        for c in range(emap.shape[1]):
            for e in range(len(entities)):
                if data[r][c][0] == entities[e]["color"][0] and data[r][c][1] == entities[e]["color"][1] and data[r][c][2] == entities[e]["color"][2]:
                    emap[r][c] = e + 1
    emaps[i] = emap
t0 = time.time()

# Global stats
total = emaps[0].shape[0] * emaps[0].shape[1]
for e in range(len(entities)):
    d0 = density = np.sum(emaps[0] == e + 1) / total
    speed = 0
    acc = 0
    for i in range(1, len(emaps)):
        d1 = np.sum(emaps[i] == e + 1) / total
        s = abs(d1 - d0)
        if speed != 0:
            a = abs(s - speed)
            acc += a
        density += d1
        speed += s
    entities[e]["meanDensity"] = density / len(emaps)
    entities[e]["meanSpeed"] = speed / len(emaps)
    entities[e]["meanAcc"] = acc / len(emaps)

print(entities)

# Convert images to subgrids
subgrids = [[[None for c in range(N)] for r in range(N)] for e in range(len(emaps))]
for e in range(len(emaps)):
    half_split = np.array_split(emaps[e], N)
    res = map(lambda x: np.array_split(x, N, axis=1), half_split)
    res = reduce(add, res)

    count = 0
    for r in range(N):
        for c in range(N):
            subgrids[e][r][c] = res[count].copy()
            count += 1

# Local stats within subgrids
total = N * N
lstatsTotal = np.zeros((3, N, N))
for e in range(len(entities)):
    lstats = np.zeros((3, N, N))   # Density, speed, acceleration
    count = 0
    for i in range(len(emaps)):
        for r in range(N):
            for c in range(N):
                ld = ls = la = 0
                # Density
                ld = np.sum(subgrids[i][r][c] == e + 1) / (subgrids[i][r][c].shape[0] * subgrids[i][r][c].shape[1])
                # Speed
                if count > 0:
                    ls = np.abs(ld - lstats[0][r][c])
                # Acceleration
                if count > 1:
                    la = np.abs(ls - lstats[1][r][c])

                lstats[0][r][c] = ld
                lstats[1][r][c] = ls
                lstats[2][r][c] = la

        count += 1
    entities[e]["lstats"] = lstats.copy()
    lstatsTotal += lstats

np.set_printoptions(precision = 2, suppress = True)
#print(entities[0]["lstats"][0])
#print(entities[0]["lstats"][1])
#print(entities[0]["lstats"][2])

fulls = [np.zeros((3, emaps[0].shape[0], emaps[0].shape[1])) for e in range(len(entities))]

h = int(emaps[0].shape[0] / N)
w = int(emaps[0].shape[1] / N)
for e in range(len(entities)):
    r = 0
    for n in range(N):
        c = 0
        for m in range(N):
            fulls[e][0][r:r+h, c:c+w] = entities[e]["lstats"][0][n][m]
            fulls[e][1][r:r+h, c:c+w] = entities[e]["lstats"][1][n][m]
            fulls[e][2][r:r+h, c:c+w] = entities[e]["lstats"][2][n][m]
            c += w-1
        r += h-1

t1 = time.time()
print(t1 - t0)

#plt.imshow(fulls[2][1], cmap = "Greys")
#plt.show()

reward = np.zeros(emaps[0].shape)
for e in range(len(entities)):
    reward[emaps[-1] == e + 1] += (1 / (entities[e]["meanDensity"])) * (0.1)
    reward += (
            #(1 - (entities[e]["meanDensity"])) \
            #+ (entities[e]["meanSpeed"]) \
            #+ (entities[e]["meanAcc"]) \
            + fulls[e][1] * 10 \
            + fulls[e][2] * 5 \
    )

norm = np.linalg.norm(reward)
if norm != 0:
    reward = reward / norm

np.savetxt(rewardOutFile, reward)
print("Saved reward grid to {}".format(rewardOutFile))

plt.imshow(reward, cmap = "Greys")
plt.show()

