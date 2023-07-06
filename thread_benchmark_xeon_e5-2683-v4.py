import numpy as np
import matplotlib.pyplot as plt

num_threads = np.arange(1, 33)
ms_per_run = np.array([
    2993, 1638, 1232, 995, 831, 727, 636, 573, 517, 469,
    434, 402, 382, 366, 364, 353, 421, 407, 391, 380,
    374, 364, 356, 351, 346, 344, 339, 336, 331, 335,
    336, 337
])

plt.bar(num_threads, 1000/ms_per_run)
plt.title("33b q4_0, Xeon E5-2683 v4, 2133 MHz quad channel RAM")
plt.xlabel("Number of threads")
plt.ylabel("Generated tokens per second")
plt.savefig("thread_benchmark_xeon_e5-2683_v4.png", dpi=360)
plt.show()
