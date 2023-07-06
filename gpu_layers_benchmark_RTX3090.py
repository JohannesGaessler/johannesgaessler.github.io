#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

MAX_LAYERS_7b = 32
MAX_LAYERS_13b = 40
MAX_LAYERS_33b = 60

eval_time_7b = np.array([
    110.77, 108.57, 106.27, 103.47, 100.35,
    97.55, 94.73, 93.07, 89.96, 87.32,
    84.80, 82.85, 79.97, 77.22, 74.41,
    72.13, 69.97, 66.65, 64.14, 61.94,
    58.66, 55.99, 53.41, 51.15, 47.73,
    45.34, 42.32, 39.92, 36.75, 33.50,
    30.78, 28.00, 25.21, 22.95
])
x_7b = np.arange(eval_time_7b.shape[0], dtype=float) / (MAX_LAYERS_7b + 1)

eval_time_13b = np.array([
    211.43, 206.91, 202.83, 198.43, 194.11,
    189.83, 185.74, 181.98, 177.12, 173.17,
    168.51, 164.62, 162.68, 155.96, 152.43,
    147.03, 144.56, 138.37, 134.92, 131.06,
    126.42, 121.78, 117.56, 113.26, 109.09,
    105.15, 100.36, 96.96, 93.04, 88.34,
    83.42, 79.43, 75.64, 70.90, 66.65,
    62.32, 57.02, 52.68, 47.92, 44.13,
    40.48, 37.42,
])
x_13b = np.arange(eval_time_13b.shape[0], dtype=float) / (MAX_LAYERS_13b + 1)

eval_time_33b = np.array([
    523.65, 515.35, 507.64, 500.95, 494.99,
    486.79, 479.01, 472.12, 466.88, 457.94,
    452.38, 444.12, 436.84, 431.36, 423.60,
    415.66, 408.67, 400.24, 393.86, 386.97,
    378.97, 370.98, 364.59, 357.85, 349.57,
    341.46, 334.93, 327.14, 320.94, 313.48,
    306.33, 298.52, 290.67, 283.58, 276.35,
    269.10, 262.31, 254.86, 246.84, 240.70,
    232.80, 225.96, 218.43, 210.80, 204.95,
    196.48, 190.29, 181.89, 174.99, 167.64,
    159.20, 151.44, 142.54, 135.74, 128.77,
    121.99, 114.17, 106.42, 99.93, 93.16,
    84.65, 81.54
])
x_33b = np.arange(eval_time_33b.shape[0], dtype=float) / MAX_LAYERS_33b

print(f"7b: {1000/eval_time_7b[0]}, {1000/eval_time_7b[-1]}, {eval_time_7b[0]/eval_time_7b[-1]}")
print(f"13b: {1000/eval_time_13b[0]}, {1000/eval_time_13b[-1]}, {eval_time_13b[0]/eval_time_13b[-1]}")
print(f"33b: {1000/eval_time_33b[0]}, {1000/eval_time_33b[-1]}, {eval_time_33b[0]/eval_time_33b[-1]}")


plt.plot(x_7b, 1000 / eval_time_7b, label="7b")
plt.plot(x_13b, 1000 / eval_time_13b, label="13b")
plt.plot(x_33b, 1000 / eval_time_33b, label="33b")
plt.legend(loc="upper left")
plt.xlabel("Proportion of GPU-accelerated layers")
plt.ylabel("Generated tokens / s")
plt.xlim(0, 1)
plt.ylim(0, 50)
plt.title("RTX 3090, Ryzen 3700X, 3200 MHz RAM")
plt.savefig("t_per_s_rtx3090.png", dpi=240)

plt.figure()

plt.plot(x_7b, eval_time_7b[0] / eval_time_7b, label="7b")
plt.plot(x_13b, eval_time_13b[0] / eval_time_13b, label="13b")
plt.plot(x_33b, eval_time_33b[0] / eval_time_33b, label="33b")
plt.legend(loc="upper left")
plt.xlabel("Proportion of GPU-accelerated layers")
plt.ylabel("Speedup")
plt.xlim(0, 1)
plt.ylim(1, 7)
plt.title("RTX 3090, Ryzen 3700X, 3200 MHz RAM")
plt.savefig("gpu_speedup_rtx3090.png", dpi=240)

plt.show()
