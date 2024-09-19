import numpy as np

i = 2
a = np.array([1, 2, 3, 4, 5])
#print the array without the ith element
print(a[np.arange(len(a))!=i])