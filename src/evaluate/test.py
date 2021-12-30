import numpy as np

a = [1,1,0,1,1,1]
b = np.array(a)
b[1:3] = list(map(lambda value: value+1 if value == 1 else value, b[1:3]))
print(b)