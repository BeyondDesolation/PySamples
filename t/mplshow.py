import numpy as np
import matplotlib.pyplot as plt

# fig, axes = plt.subplots(2, 2)
# ax2 = fig.add_subplot(1, 1, 1)
fig, ax = plt.subplots()
x = np.arange(30)
y = np.random.normal(10, 3, 30)

print(x, y)

ax.plot(x, y)
plt.show()

