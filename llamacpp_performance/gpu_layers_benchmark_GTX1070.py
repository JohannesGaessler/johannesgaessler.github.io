#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

MAX_LAYERS_7b = 32
MAX_LAYERS_13b = 40
MAX_LAYERS_33b = 60

eval_time_7b = np.array([
    109.31, 107.93, 107.46, 105.83, 105.92, 104.44, 103.57, 102.19, 101.30, 101.63,
    99.63, 99.12, 98.98, 97.22, 95.81, 95.10, 93.81, 93.04, 92.20, 91.42,
    90.39, 89.24, 88.50, 87.15, 86.24, 85.49, 85.00, 83.96, 82.82, 82.42,
    81.87, 80.91, 80.03
])
x_7b = np.arange(eval_time_7b.shape[0], dtype=float) / MAX_LAYERS_7b

eval_time_13b = np.array([
    205.94, 205.74, 203.19, 201.61, 200.78, 198.51, 197.02, 195.08, 194.17, 192.45,
    190.47, 188.73, 187.93, 186.19, 183.87, 182.66, 180.59, 179.92, 177.75, 176.91,
    175.13, 173.60, 171.54, 169.32, 168.33, 167.56, 165.78, 163.82, 162.33, 160.97,
    159.66, 158.64, 157.60, 156.57, 155.70,
])
x_13b = np.arange(eval_time_13b.shape[0], dtype=float) / MAX_LAYERS_13b

eval_time_33b = np.array([
    503.56, 501.11, 498.14, 496.44, 492.99, 488.95, 488.56, 484.59, 482.11, 478.47,
    477.80, 474.87, 470.92, 468.16, 464.54, 463.73, 459.93, 456.00, 452.93, 450.53,
])
x_33b = np.arange(eval_time_33b.shape[0], dtype=float) / MAX_LAYERS_33b

print(f"7b: {1000/eval_time_7b[0]}, {1000/eval_time_7b[-1]}, {eval_time_7b[0]/eval_time_7b[-1]}")
print(f"13b: {1000/eval_time_13b[0]}, {1000/eval_time_13b[-1]}, {eval_time_13b[0]/eval_time_13b[-1]}")
print(f"33b: {1000/eval_time_33b[0]}, {1000/eval_time_33b[-1]}, {eval_time_33b[0]/eval_time_33b[-1]}")


plt.plot(x_7b, eval_time_7b[0] / eval_time_7b, label="7b")
plt.plot(x_13b, eval_time_13b[0] / eval_time_13b, label="13b")
plt.plot(x_33b, eval_time_33b[0] / eval_time_33b, label="33b")
plt.legend(loc="upper left")
plt.xlabel("Proportion of GPU-accelerated layers")
plt.ylabel("Speedup")
plt.xlim(0, 1)
plt.ylim(1, 1.5)
plt.title("GTX 1070, Ryzen 3700X, 3200 MHz RAM")
plt.savefig("gpu_speedup_gtx1070.png", dpi=240)
plt.show()
