import numpy as np
from calculator import hm
a = [98.69, 94.36,96.47,85.07,60.84,70.94,95.33,78.25,85.94,48.46,37.35,42.11,90.52,91.66,91.08,78.94,69.93,74.16,98.67,77.71,86.94,95.53,96.89,96.21,83.54,75.32,79.22,83.05,79.01,80.98,89.09,79.07,83.78]

base = a[::3]
new = a[1::3]
hm_accs = [hm(base[i], new[i]) for i in range(len(base))]
for i in range(len(hm_accs)):
    print(f"{base[i]:.2f}: {new[i]:.2f}: {hm_accs[i]:.2f}")
print(f"base: {np.mean(base):.2f}")
print(f"new: {np.mean(new):.2f}")
print(f"hm: {np.mean(hm_accs):.2f}")