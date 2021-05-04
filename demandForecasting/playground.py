import numpy as np

a = np.arange(72).reshape(8,9)

print(a)

# b = a.reshape(-1,4,a.shape[-1]).sum(1)
b = a.reshape(-1,4,a.shape[-1])

c = b[1,0,0]
print(b)
print(c)