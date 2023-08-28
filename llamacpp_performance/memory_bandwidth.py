import numpy as np
import matplotlib.pyplot as plt

memory_speeds = np.array([1600, 1866, 2133, 2400, 2666, 2733, 2800, 2866, 2933, 3000, 3066, 3133, 3200, 3266, 3333, 3400, 3466])
ms_per_run = np.array([944, 814, 715, 640, 577, 567, 553, 542, 531, 521, 510, 501, 491, 484, 474, 466, 457])
tokens_per_second = 1 / (ms_per_run / 1000)

plt.plot([0, 3*memory_speeds[0]], [0, 3*tokens_per_second[0]], label="Proportional scaling")
plt.plot(memory_speeds, tokens_per_second, label="Benchmark")
plt.legend(loc="upper left")
plt.xlabel("Dual channel memory speed [MHz]")
plt.ylabel("Generated tokens per second")
plt.title("LLaMa 33b q4_0, Ryzen 3700X, 5 threads")
plt.xlim(memory_speeds[0], memory_speeds[-1])
plt.ylim(1, 2.5)
plt.savefig("memory_scaling_1.png", dpi=360)
plt.figure()
plt.xlabel("Memory speed [MHz]")
plt.ylabel("Tokens per second per GHz memory speed")
plt.plot(memory_speeds, tokens_per_second / (memory_speeds / 1000))
plt.savefig("memory_scaling_2.png", dpi=360)
plt.show()
