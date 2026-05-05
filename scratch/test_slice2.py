import ctypes
Arr = ctypes.c_int * 10
arr = Arr()
try:
    arr[2:3] = 42
    print("Assigned int to slice of length 1")
except Exception as e:
    print("Exception int length 1:", repr(e))

try:
    arr[2:3] = [42]
    print("Assigned list to slice of length 1")
except Exception as e:
    print("Exception list length 1:", repr(e))
