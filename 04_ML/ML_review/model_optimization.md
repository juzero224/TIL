# 머신러닝 모델 최적화
## 주성분 분석 (PCA, Principal Component Analysis)
- 주성분 분석(PCA)란?
  - 고차원 데이터 집합이 주어졌을 때 원래의 고차원 데이터와 가장 비슷하면서 더 낮은 차원 데이터를 찾아내는 방법
- **변수 선택** 및 **차원 축소** 방법
- 주성분 분석은 상관관계가 있는 변수드을 선형결합(Linear combination)해서 분산이 극대화된 상관관계가 없는 새로운 변수(주성분)들로 축약하는 것

- 통계 데이터 분석(주성분 찾기), 데이터 압축, 노이즈 제거 등 여러 분야에 사용

$$
c = \begin{bmatrix}cov(x,x) & cov(x,y) \\ cov(x,y) & cov(y,y) \end{bmatrix} = \begin{bmatrix} \frac{1}{n}\sum(x_i-m_i)^2 & \frac{1}{n} \sum(x_i-m_x)(y_i-m_y) \\ \frac{1}{n}\sum(x_i-m_x)(y_i-m_y) & \frac{1}{n}\sum(y_i-m_y)^2 \end{bmatrix}  
$$

```python
sklearn.decomposition.PCA(n_components=None, copy=True,
whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto',
random_state=None)
```

```python
import seaborn as sns
from sklearn.decomposition import PCA
iris = sns.load_dataset('iris')
iris_X, iris_y = iris.iloc[:,:-1], iris.species

pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_X)
```
```python
iris_pca[:5,:]
```
> array([[-2.68412563,  0.31939725],
>       [-2.71414169, -0.17700123],
>       [-2.88899057, -0.14494943],
>       [-2.74534286, -0.31829898],
>       [-2.72871654,  0.32675451] ])

```python
pca.explained_variance_
# 각각의 주성분 벡터가 이루는 축에 투영(projection) 한 결과의 분산
```
> array([4.22824171, 0.24267075])

```python
pca.components_
```
> array([[ 0.36138659, -0.08452251,  0.85667061,  0.3582892 ],
       [ 0.65658877,  0.73016143, -0.17337266, -0.07548102] ])

