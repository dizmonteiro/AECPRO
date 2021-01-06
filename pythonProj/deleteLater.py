import numpy as np

p1 = np.array([2, 6])
p2 = np.array([10, 2])
p3 = np.array([2.5, 2.5])

# vector p2p1
print(p2-p1)
# vector p3p1
print(p3-p1)
# produto
print(np.cross(p2-p1,p3-p1))
# norma
print(np.linalg.norm(p2-p1))

print(abs(np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1)))

# https://www.qc.edu.hk/math/Advanced%20Level/Point_to_line.htm
# - ver method 2