# 데이터 전처리 (Preprocessing)

코드 >> [ML_02. Data_preprocessing](https://github.com/juzero224/TIL/blob/a848f7a5a86d1af9c47592a16db7298af80e4535/04_ML/Multi_ML_Python/ML_02.%20Data_preprocessing.ipynb)



- 결손데이터(Missing Data 처리) - NaN/Null 처리

- 데이터 인코딩 - 레이블 인코딩(label encoding), 원-핫인코딩(One-Hot encoding)
  - 문자열, 카테고리형 >> 숫자형
  - 텍스형 >> 벡터화하거나 삭제
- 피처 스케일링(Feature Scaling)과 정규화
  - 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
    - 표준화(Standardization), 정규화(Normalization)
  - 이상치 제거
  - Feature 선택, 추출 및 가공 -> Feature Engineering

<br>

<br>

-----------------------

<br>

<br>

## 1. 결손 데이터 처리 - NaN/Null 처리  ==SimpleImputer==

- 사이킷런의 `SimpleImputer` 클래스, 판다스의 `isnan()` -> `fillna()` or `dropna()`
- strategy : string, optional (default = 'mean')
- `strategy = 'mean'` : 평균값 (default)
- `strategy = 'median'` : 중앙값
- `strategy = 'most_frequent'` : 최빈값 (mode)으로 대치
- `strategy = 'constant', fill_value=1` : 특정값으로 대치



- SimpleImputer NaN처리 Python

```python
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(strategy='mean')  # nan >> mean 평균값으로 대치 (단순대입법)
imp_mean = imp_mean.fit(x[:,1:3])
x[:,1:3] = imp_mean.transform(x[:,1:3]) # scailing한 값으로 대체

'''
nan >> 평균값으로 대치
'''
```

<br>

<br>

------------------

<br>

<br>



## 2. 피처 스케일링(Feature Scaling)과 정규화

<br>

### 표준화(standardization)   ==StandardScaler==

- 값의 분포를 평균 0이고, 분산이 1인 가우시안 정규분포를 가진 값 -> `standardScaler`
- 경사하강법(서포트 벡터머신이나, 선형회귀, 로지스틱 회귀)에서 데이터가 정규분포를 가지고 있다고 가정하고 구현됐기 떄문에 표준화를 적용하는 것은 예측 성능 향상에 중요한 요소
- mu는 한 특성의 평균값이고 sigma는 표준편차
- 어떤 특성의 값들이 정규분포를 따른다고 가정하고 평균 0, 표준편차 1을 갖도록 변환해 주는 것
- 최솟값과 최댓값의 크기를 제한하지 않기 때무에 이상치를 파악할 수 있음
- 정규화처럼 특성값의 범위가 0과 1의 범위로 균일하게 바뀌지 않음

$$
\frac{xi-mean(x)}{stdev(x)}\\
\frac{X-\mu}{\sigma}
$$

- 표준화 Python

```python
# 표준화
from sklearn.preprocessing import StandardScaler # 표준화 지원 클래스

sc_x = StandardScaler()
sc_x.fit_transform(x[:,1:3])
x[:,1:3] = sc_x.fit_transform(x[:,1:3])
x
```

<br>

<br>

<br>

### 정규화(Normalization)   ==MinmaxScaler==

- 서로 다른 피처의 크기를 통일하기 위해 크기를 변환해 주는 개념 -> `MinMaxScaler`
- 최소 0 ~ 최대 1
- 데이터 분포가 정규 분포가 아닐 경우 적용
- Nomalization > MinMaxScaler

$$
\frac{xi-min(x)}{max(x)-min(x)}
$$

<br>

| Normalization                            | Standardization                            |
| ---------------------------------------- | ------------------------------------------ |
| 스케일링 시 최대, 최소값이 사용된다      | 스케일링 시 평균과 표준편차가 사용된다     |
| 피처의 크기가 다를 때 사용               | 평균이 0, 표준편차가 1로 만들 떄 사용      |
| [0,1] (또는 [-1,1]) 사이 값으로 스케일링 | 특정 범위로 제한되지 않음                  |
| 분포에 대해 모를 때 유용                 | 피처가 정규분포(가우시안 분포)인 경우 유용 |
| MinMaxScaler, Normalizer                 | StandardScaler, RobustScaler               |

<br>

- MinMaxScaler Python

```python
# 정규화 min_max 최소-최대
from sklearn.preprocessing import MinMaxScaler # 정규화 지원 모듈

sc_x_minmax = MinMaxScaler()
sc_x_minmax.fit_transform(x[:,1:3])
x[:,1:3] = sc_x_minmax.fit_transform(x[:,1:3])
x
```



 







______

참고 자료

천방지축 Tech 일기장, [통계] 정규화 vs 표준화

https://heeya-stupidbutstudying.tistory.com/32