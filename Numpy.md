---
sticker: emoji//2139-fe0f
python lib:
---

---
## Basic Numpy
``` python
import micropip
await micropip.install("numpy")
```
```python
import numpy as np
print(np.array([1, 2, 3])) #prints array
```
```python
a= np.array([5, 2, 3])
print(a[0], a[1], a[0:], a[1:3], a[1:-1], a[::2])
print(a[[0, 1,2]])
```

## Array Types
```python
a = np.array([1, 2, 3, 4])
print(a.dtype) #int64 = 64 bits
float_array = np.array([1, 2, 3, 4], dtype=np.float64) 
print(float_array) #integers are converted to floating-point numbers (with decimal points)
```
```python
c = np.array(['a', 'b', 'c']) 
print(c.dtype) # '<U1' means Unicode string with 1 character 
# The '<' indicates little-endian byte ordering
# `<U5` for strings of length 5)
# fixed-length Unicode strings for efficiency
```
```python
import sys
d = np.array([{'a': 1}, sys]) 
print(d) 
print(d.dtype) #'O' stands for Object - these arrays store references to objects
#dictionary, module are the elements
```

## Dimensions and Shapes

```python
B = np.array([
    [
        [12, 11, 10],
        [9, 8, 7],
    ],
    [
        [6, 5, 4],
        [3, 2, 1]
    ]
])
print(B.shape) #matrices ,rows, columns
print(B.ndim) #no. of dimensions
```
<mark style="background: #FFB8EBA6;">If the shape isn't consistent, it'll just fall back to regular Python objects
</mark>
```python
C = np.array([
    [
        [12, 11, 10],
        [9, 8, 7],
    ],
    [
        [6, 5, 4]
    ]
])
print(C.dtype) # 'O'
print(C.shape) # 2 bec 2 ele in list
print(C.size) # 2
print(type(C[0]))
```
## Indexing and Slicing of Matrices
```python
A = np.array([
#    0  1  2  <- column indices
    [1, 2, 3], # 0 <- row index
    [4, 5, 6], # 1
    [7, 8, 9]  # 2
])
```
```python
row_1 = A[1]  # Gets the entire row at index 1
print(row_1)  # Output: array([4, 5, 6])

element_1_0 = A[1][0]  # Gets element at row 1, column 0
print(element_1_0)  # Output: 4

element_1_0_comma = A[1, 0]  # Gets element at row 1, column 0
print(element_1_0_comma)  # Output: 4
# A[d1, d2, d3, d4] is the general format for indexing n-dimensional arrays
```
```python
first_two_rows = A[0:2]  # Rows with indices 0 and 1 (2 is exclusive)
print(first_two_rows)
# array([[1, 2, 3],
#        [4, 5, 6]])

first_two_cols = A[:, :2]  # All rows, columns with indices 0 and 1
print(first_two_cols)
# array([[1, 2],
#        [4, 5],
#        [7, 8]])

top_left = A[:2, :2]  # First two rows, first two columns
print(top_left)
# array([[1, 2],
#        [4, 5]])

last_col_first_two_rows = A[:2, 2:]  # First two rows, column index 2 and beyond
print(last_col_first_two_rows)
# array([[3],
#        [6]])
```
```python
A[1] = np.array([10, 10, 10])  # Replace entire row at index 1
# array([[ 1,  2,  3],
#        [10, 10, 10],
#        [ 7,  8,  9]])

# Broadcasting: set an entire row to a single value
A[2] = 99  # Replace entire row at index 2
# array([[ 1,  2,  3],
#        [10, 10, 10],
#        [99, 99, 99]])

```

## Math Operations
```python
a = np.array([1, 2, 3, 4])

a.sum(), a.mean(), a.std() , a.var()
# 10 , 2.4, 1.118033988749895, 1.25
```
```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

A.sum(), A.mean(), A.std()
# 45 , 5.0 , 2.581988897471611

A.sum(axis=0) # sum of columns
# array([12, 15, 18])

A.sum(axis=1) # sum of rows
# array([ 6, 15, 24])

A.mean(axis=0), A.mean(axis=1)
# array([4., 5., 6.]), array([2., 5., 8.])

```

## Broadcasting and Vectorized Operations
```python
a = np.arange(4)
# array([0, 1, 2, 3])
a + 10
# array([10, 11, 12, 13])
a * 10
# array([ 0, 10, 20, 30])
a
# array([0, 1, 2, 3])
a += 100
a
# array([100, 101, 102, 103])
```
```python
l = [0, 1, 2, 3]
[i * 10 for i in l]
# [0, 10, 20, 30]

a = np.arange(4)
# array([0, 1, 2, 3])

b = np.array([10, 10, 10, 10])
# array([10, 10, 10, 10])

a + b
# array([10, 11, 12, 13])

a * b
# array([ 0, 10, 20, 30])
```

## Boolean Arrays 
```python
a = np.arange(4)
# array([0, 1, 2, 3])
a[0], a[-1]
# (0, 3)
a[[0, -1]]
# array([0, 3])

a[[True, False, False, True]]
# array([0, 3]) 0 and 3 are the VALUES

a
# array([0, 1, 2, 3])

a >= 2 #which one's are >=2
# array([False, False, True, True])

a[a >= 2] #a[false,false,true,true]
# array([2, 3])

a.mean()
# 1.5
a[a > a.mean()]
# array([2, 3]) 2 AND 3 ARE THE VALUES

a[~(a > a.mean())]
# array([0, 1])

a[(a == 0) | (a == 1)]
# array([0, 1])

a[(a <= 2) & (a % 2 == 0)]
# array([0, 2])
```
```python
A = np.random.randint(100, size=(3, 3))

A
# array([[71,  6, 42],
#        [40, 94, 24],
#        [ 2, 85, 36]])

A[np.array([
   [True, False, True],
   [False, True, False],
   [True, False, True]
])]
# array([71, 42, 94, 2, 36])

A > 30
# array([[ True, False,  True],
#        [ True,  True, False],
#        [False,  True,  True]])

A[A > 30]
# array([71, 42, 40, 94, 85, 36])
```

## Linear Algebra
```python
A = np.array([
   [1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]
])

B = np.array([
   [6, 5],
   [4, 3],
   [2, 1]
])

A.dot(B) # or 
A @ B
# array([[20, 14],
#        [56, 41],
#        [92, 68]])

B.T #transpose
# array([[6, 4, 2],
#        [5, 3, 1]])
```
## Extra Functions
### Random number generation functions
```python
np.random.random(size=2) # arr of 2 random floats between 0 and 1
np.random.normal(size=2) # arr of 2 random numbers from std normal dist 
#(mean=0, stddev=1) 
np.random.rand(2, 4) # Generate 2x4 array of random floats between 0 and 1
```
### arange: evenly spaced values
```python
np.arange(10) # Create array of integers from 0 to 9 
np.arange(5, 10) # Create array of integers from 5 to 9 
print(np.arange(0, 1, .1))# Create array from 0 to 0.9 with step 0.1
```
### reshape: change the shape of an array but not data
```python
print(np.arange(10).reshape(2, 5)) # array 0-9 and reshape to 2 rows, 5 columns
```
### linspace: create evenly spaced numbers over a specified interval
<mark style="background: #FFB8EBA6;">can be inclusive</mark>
```python
np.linspace(0, 1, 5) #5 evenly spaced numbers from 0 to 1 (inclusive) 
np.linspace(0, 1, 20) #20 evenly spaced numbers from 0 to 1 (inclusive) 
np.linspace(0, 1, 20, False) #20 evenly spaced numbers from 0 to 1 (excluding endpoint)
```
### zeros, ones, empty: specified shape and type
```python
np.zeros(5) #1D array of 5 zeros (float by default) 
np.zeros((3, 3)) #3x3 array of zeros 
print("zero")
print(np.zeros((3, 3), dtype=np.int64)) # Create 3x3 array of zeros with integer data type 
print("ones")
print(np.ones(5)) #1D array of 5 ones 
print("ones")
print(np.ones((3, 3))) #3x3 array of ones 
np.empty(5) #1D array of 5 uninitialized values (values depend on memory state) 
print("empty")
print(np.empty((2, 2))) #2x2 array of uninitialized values
```
### identity matrices
```python
print(np.identity(2))
```
```python
print(np.eye(3,3)) #ones on the diagonal on the main diagonal
```
```python
print(np.eye(3,3,k=1)) #ones on the diagonal above the main diagonal 
#could be neg too for below
```
