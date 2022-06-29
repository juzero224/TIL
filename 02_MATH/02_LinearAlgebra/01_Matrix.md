# 선형대수



## 벡터

- 벡터 = 리스트(list)로 구현

```python
def v_add(u,v):
    '''
    벡터의 덧셈
    입력값 : 더하고자 하는 벡터 u, v
    출력값 : 벡터 u, v의 덧셉 결과 w
    '''
    n = len(u)
    w = []
    
    for i in range(0,n):
        val = u[i] + v[i]
        w.append(val)
    return w
```

> > i=0 : val 1+4 >> w=[5]
> >
> > i=1 : val 2+5 >> w=[5,7]
> >
> > i=2 : val 3+6 >> w=[5,7,9]



1. 행벡터와 열벡터의 곱은 **스칼라**

2. 열벡터와 행벡터의 곱은 **행렬**
3. 행렬과 열벡터의 곱은 **열벡터**
4. 행벡터와 행렬의 곱은 **행벡터**





--------------





## 행렬

### 덧셈

- 행렬의 덧셈은 같은 차수를 가지는 행렬에 대해서만 적용
- (A+B)' = A' + B'
- tr(A+B) = tr(A) + tr(B)   (tr : trace of sum ( 합에 대한 궤적))

$$
tr(A+B) = \sum_{i=1}^n (a_{ii}+b_{ii}) = \sum_{i=1}^n a_{ii} + \sum_{i=1}^n b_{ii} = tr(A) + tr(B)
$$



```python
def add(A,B):
    '''
    행렬의 덧셈
    입력값 : 행렬의 덧셉을 수행할 행렬 A, B
    출력값 : 행렬 A와 행렬 B의 덧셈 결과인 행렬 res
    '''
    
    n = len(A)
    p = len(A[0])
    
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] + B[i][j]
            row.append(val)
        res.append(row)
    return res
```





### 행렬 곱


$$
A_{r \times c} B_{c \times s} = a'_i b_j
                              = a_{i1}b_{1j} + a_{i2}b_{2j} + ... + a_{ic}b_{cj}\\
                              = \sum_{k=1}^c a_{ik}b_{kj}
$$

$$
 \begin{bmatrix}i번째 행 \\\rightarrow  \end{bmatrix}_{r\times c} \begin{bmatrix}\downarrow & j번째 열 \ \end{bmatrix}_{c\times s} =  \begin{bmatrix}(i,j)번째 원소 \end{bmatrix}_{r \times s}
$$



```python
def matmul(A,B):
    '''
    행렬의 행렬곱
    입력값 : 행렬곱을 수행할 행렬 A, B
    출력값 : 행렬 A와 행렬 B의 행렬곱 결과인 행렬 res
    '''
    
    n = len(A)
    p1 = len(A[0])
    p2 = len(B[0])
    
    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p2):
            val = 0
            for k in range(0, p1):
            	val += A[i][j] * B[i][j]
            row.append(val)
        res.append(row)
    return res
```

```python
A = [[2,7],[3,4],[5,2]]
B = [[3,-3,5],[-1,2,-1]]
matmul(A,B)
```



### 전치 행렬

- A^T
- (A')' = A



```python
def matmul(A,B):
    '''
    행렬의 전치행렬
    입력값 : 전치행렬을 구하고자 하는 행렬 A
    출력값 : 행렬 A의 전치행렬 At
    '''
    
    n = len(A)
    p = len(A[0])
    
    At = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[j][i]
            row.append(val)
        At.append(row)
    return At
```

```python
import numpy as np
A = np.array([[1,5],[3,4],[6,2]])
At = np.transpose(A)
```

```python
At = A.T
```



### 대각 행렬

- 정방행렬에서 a11.a22, ... , a_nn : 대각 원소
- 대각 원소 외의 원소를 비대각 원소라 하는데, 비대각 원소 =0 >> 대각 행렬

- `AD` >> 열에 영향
- `DA` >> 행에 영향



### 항등행렬

$$
I_p A_{p \times q} = A_{p \times q} I_p = A_{p \times q}
$$



```python
def identity(n):
    '''
    항등행렬 생성
    입력값 : 항등 행렬의 크기 n
    출력값 : nxn 항등 행렬 I
    '''
    I = []
    for i in range(0,n):
        row = []
        for j in range(0,n):
            if i==j:
                row.append(1)
            else:
                row.append(0)
        I.append(row)
    return I
```

```python
import numpy as np
I = np.identity(5)
```

```python
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
I = np.identity(3)
AI = np.matmul(A,I)
IA = np.matmul(I,A)
```



### 영 행렬

```python
def zero_mat(n,p):
    '''
    영 행렬 생성
    입력값 : 생성하고자 할 영 행렬의 크기  n행, p열
    출력값 : nxp 영 행렬 Z
    '''
    Z = []
    for i in range(0,n):
        tmp_row = []
        for j in range(0,p):
            tmp_row.append(0)
        Z.append(tmp_row)
    return Z
```

```python
Z = np.zeros((3,2))
```



### 삼각 행렬

```python
def up_tri(A):
    '''
    상 삼각 행렬 변환
    입력값 : 상 삼각 행렬로 변환하고자 하는 행렬 A
    출력값 : 행럴 A를 상 삼각 행렬로 변환시킨 행렬 up_tri
    '''
    n = len(A)
    p = len(A[0])
    uptri = []
    
    for i in range(0,n):
        row = []
        for j in rnage(0,p):
            if i>j:
                row.append(0)
            else:
                row.append(A[i][j])
        uptri.append(row)
    return uptri
```

```python
Au = np.triu(A)
```



### 멱등행렬

$$
M \neq I,\ M \neq 0\\
M^2 = M
$$

을 만족하는 M을 멱등행렬 (idempotent matrix)





### 토플리츠 행렬

- 시계열 분석시 사용 -> LSTM

$$
A = \begin{bmatrix}a & b & c & d & e \\f & a &b & c & d \\ g & f & a & b & c \\ h & g & f & a & b \\ i & h & g & f & a \end{bmatrix} 
$$



### 이중 대각 행렬

- upper bidiagonal matrix
  - 대각행렬이 2줄


$$
A = \begin{bmatrix}1 & 3 & 0 & 0 \\0 & 2 & 2 & 0 \\ 0 & 0 & 3 & 5 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$



- lower bidiagonal matrix



--------------------------------

## 직교행렬

- 직교행렬 (orthogonal matrix)
  - P는 정방행렬

$$
PP' = P'P = I
$$



- 실수 벡터 x' = [x1 x2 ... xn]

$$
x의\ norm = ||x|| = \sqrt{x'x} = (\sum_{i=1}^n x_i^2)^{\frac{1}{2}}
$$

- norm이 1인 벡터를 **정규벡터**

$$
u = (\frac{1}{\sqrt{x'x}}x)
$$



### 하우스홀더 행렬

- 정방행렬을 삼각화하는 데 유용한 직교행렬
- 벡터와 벡터의 연산으로 V.T 뒤에 붙으면 행렬  >> VV^T : 행렬
- V.T 앞에 붙으면 스칼라 만드는 행렬 >> V^TV : 스칼라

$$
v =  \begin{bmatrix} v_1 \\ v_2 \\  \vdots  \\ v_n\end{bmatrix} \\
H = I - 2 \frac{VV^T}{V^TV}
$$





-------------------------------

plus study



## 2차 형식

$$
\sum_{i=1}^n (x_i-\bar{x})^2 = x'Cx
$$

- 통계의 분산분석의 일반적 이론에서 많이 쓰임





## 양정치 행렬

- 양정치 행렬 (positive definite)
  - x != 0일 떄, 2차 형식 x'Ax가 항상 양수가 되는 2차 형식

$$
x = 0이\ 아닌\ 모든\ x에\ 대하여\ x'Ax >0
$$

- 양반정치(positive semidefinite)

$$
모든\ x에 대해\ x'Ax \geq 0\ 그리고\ 어떤\ x \neq 0에 대해 x'Ax = 0
$$





-------



참고 자료

- 닥터윌, AI ML/DL를 위한 선형대수 with 파이썬 1장
- 통계학을 위한 행렬대수학, 자유아카데미, 1-3장
