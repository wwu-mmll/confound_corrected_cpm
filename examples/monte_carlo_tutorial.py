import numpy as np


assoc_strength = 0.5

x = np.random.normal(0, 1, 1000)
e = np.random.normal(0, 1, 1000)
y = assoc_strength * x + e

print(f"var({assoc_strength}*x)={np.var(assoc_strength * x)}")
print(f"var(noise)={np.var(e)}")
print(f"var(y)={np.var(y)}")

print(f"r={np.corrcoef(x, y)[0, 1]:.2f}")
print(f"Explained variance: {np.square(np.corrcoef(x, y)[0, 1]):.2f}")