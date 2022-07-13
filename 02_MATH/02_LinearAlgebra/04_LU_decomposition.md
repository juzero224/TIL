## LU 분해

기본 행렬의 역행렬을 곱하기


$$
A=LU\\
\begin{bmatrix}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{bmatrix}=\begin{bmatrix}l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33}\end{bmatrix}\begin{bmatrix}u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33}\end{bmatrix}
$$
행렬 A를 하삼각행렬과 상삼각행렬의 곱으로 나타내줄 수 있다





-------------------------

> 풀이

- Row multiplication 행렬과 역행렬

$$
E=\begin{bmatrix}1 & 0 & 0 \\ 0 & \color{red}{s} & 0 \\ 0 & 0 & 1\end{bmatrix} \rightarrow E^{-1}=\begin{bmatrix}1 & 0 & 0 \\ 0 & \color{red}{1/s} & 0 \\ 0 & 0 & 1\end{bmatrix}
$$

- Row addition 행렬과 역행렬

$$
E=\begin{bmatrix}1 & 0 & 0 \\ \color{red}{s} & 1 & 0 \\ 0 & 0 & 1\end{bmatrix}\rightarrow E^{-1}=\begin{bmatrix}1 & 0 & 0 \\ \color{red}{-s} & 1 & 0 \\ 0 & 0 & 1\end{bmatrix}
$$

- 행의 순서를 바꿔주는 기능을 수행하는 기본 행렬과 역행렬

$$
P_{31}=\begin{bmatrix}0 & 0& \color{red}{1} \\ 0 & 1 & 0 \\ \color{blue}{1} & 0 & 0\end{bmatrix} \rightarrow P^{-1}_{31}=\begin{bmatrix}0 & 0& \color{red}{1} \\ 0 & 1 & 0 \\ \color{blue}{1} & 0 & 0\end{bmatrix}
$$

- 기본 행렬에 역행렬 곱해주기

$$
E_nE_{n-1}...E_2E_1A = U
$$

$$
A = E_1^{-1}E_2^{-1}...E_{n-1}^{-1}E_n^{-1}U
$$

$$
E_1^{-1}E_2^{-1}...E_{n-1}^{-1}E_n^{-1} = L
$$

------------------

### Ax=b 해 구하기

A가 정방행렬이고 A=LU와 같이 분해 가능하다면 
$$
Ax = b
$$
에서 A = LU로 바꿀 수 있으므로
$$
LUx = b
$$

$$
Ax = b \Leftrightarrow LUx=b
$$

이 식에서 괄호의 위치를 바꿔보면
$$
L(Ux)=b
$$
Ux = 일종의 열벡터

따라서, Ux = y로 치환하면
$$
Ly=b
$$
L은 하삼각행렬
$$
Ux = y
$$
Ux=y를 앞서 구한 y값에 넣고 풀면 x에 대한 답을 얻을 수 있다





------------

### 행렬식 수월하게 구하기

$$
A = LU
$$

라고 했을 때, 행렬식의 성질에 의해
$$
det(A) = det(L)det(U)
$$
라고 할 수 있다.

L과 U는 모두 삼각행렬 >> 대각성분 곱으로 행렬식 계산 가능
$$
det(A) = \prod_{i=1}^n l_{i,i}\prod_{j=1}^n u_{j,j}=\prod_{i=1}^n l_{i,i}u_{i,i} 
$$


------------

### 파이썬

```python
A = [[2,-2,-2],[0,-2,2],[-1,5,2]]

L, U = lu_decomp(A)
print(L)
print(U)
```

```python
import numpy as np
from scipy.linalg import lu

A = np.array([[2,-2,-2],[0,-2,2],[-1,5,2]])
P, L, U = lu(A)

print(L)
print(U)
```



