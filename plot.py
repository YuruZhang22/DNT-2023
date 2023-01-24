import imp
import matplotlib.pyplot as plt
import matplotlib

dataset = [500, 1000, 2000, 3000, 3888]
difference = [0.179726, 0.1671113, 0.1612487, 0.170485, 0.1643444]

plt.xlabel('(scale = 0.2)')

plt.plot(dataset, difference)
plt.show()

scale = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
difference = [0.167448, 0.17673, 0.170069, 0.1612487, 0.167218, 0.170227]

plt.xlabel('(dataset = 2000)')

plt.plot(scale, difference)
plt.show()