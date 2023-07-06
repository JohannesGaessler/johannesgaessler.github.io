import numpy as np
import matplotlib.pyplot as plt

num_threads = np.arange(1, 17)
ms_per_run = np.array([1599, 823, 584, 496, 489, 494, 500, 505, 548, 538, 551, 545, 548, 549, 556, 563])

plt.bar(num_threads, 1000/ms_per_run)
plt.title("33b q4_0, Ryzen 3700X, 3466 MHz dual channel RAM")
plt.xlabel("Number of threads")
plt.ylabel("Generated tokens per second")
plt.savefig("thread_benchmark_rx_3700x.png", dpi=360)
plt.show()
