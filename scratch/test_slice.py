import ctypes
import numpy as np

Arr = ctypes.c_int * 10
arr = Arr()
try:
    arr[2:5] = 42
    print("Assigned int to slice")
except Exception as e:
    print("Exception int:", repr(e))

try:
    arr[2:5] = [42] * 3
    print("Assigned list to slice")
except Exception as e:
    print("Exception list:", repr(e))

