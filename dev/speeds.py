import matplotlib.pyplot as plt
import numpy as np

def squelch(x):
    a, b  = 0.3, 5
    return np.where(x>a, x, a + b*(x-a))

fig, axs = plt.subplots()
x = np.array(range(25))/25
axs.plot(x, squelch(x))
axs.set_ylim(0,1)
plt.show()
