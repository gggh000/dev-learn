import matplotlib.pyplot as plt
import numpy as np

plt.get_backend()

x = np.linspace(-10, 10, 100)
y = np.sin(x)

plt.plot(x,y, marker="x")
plt.show()