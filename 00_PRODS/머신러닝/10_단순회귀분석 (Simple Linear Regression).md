# 10_단순회귀분석 (Simple Linear Regression)



- Simple Linear Regression : 특성변수 X 하나만 가지고 연속형 종속변수 Y를 예측하는 모델



## 회귀분석 개요

### 회귀분석

- 독립변수와 종속변수 간의 함수적인 관련성을 규명하기 위하여 어떤 수학적 모형을 가정하고, 이 모형을 측정된 자료로부터 통계적으로 추정하는 분석방법



- y = f(x)의 함수 관계가 있을 때,
  - x를 설명변수(explanatory variable) 또는 독립변수 (independent variable), 예측, 특성
    - 단순 회귀 : 독립변수 1개
    - 다중 회귀 : 독립변수가 2개 이상
  - y를 반응변수(response variable) 또는 종속변수 (dependent variable), 목표
- 직선의 형태
  - y ≈ a + bx



## 단순선형회귀모형

### 모형 정의 및 가정

- 자료(𝑥𝑖, 𝑌𝑖), 𝑖=1,⋯, 𝑛 에 다음의 관계식이 성립한다고 가정함.
  $$
  Y_i = \alpha + \beta x_i + \varepsilon_i \ ,i= 1,2,...,n
  $$

  - 오차항인 𝜀1, 𝜀2, ⋯, 𝜀𝑛 는서로 독립인 확률변수로, **𝜀𝑖~𝑁[0,𝜎^2]** 

    : **정규, 등분산, 독립가정**

  - α,𝛽는 회귀계수라 부르며 α는 절편, 𝛽은 기울기를 나타냄.
  - α,𝛽, 𝜎^2은 미지의 모수로, 상수

![image-20221028224613712](C:/Users/yes47/AppData/Roaming/Typora/typora-user-images/image-20221028224613712.png)



## 단순선형회귀모형의 모수 추정

### 모수 추정

- 모형이 포함한 미지의 모수  α, 𝛽를 추정하기 위하여 각 독립변수 x_i에 대응하는 종속변수 y_i로 짝지어진 n개의 표본인 **관측치 (x_i, y_i)** 가 주어짐

  ![image-20221028224834745](C:/Users/yes47/AppData/Roaming/Typora/typora-user-images/image-20221028224834745.png)



### 최소제곱법

- 단순회귀모형 𝑌𝑖 = 𝛼 + 𝛽𝑥𝑖 + 𝜀𝑖 에서 자료점과 회귀선 간의 **수직거리 제곱합**
  $$
  SS(\alpha, \beta) = \sum_{i=1}^n (y_i - \alpha - \beta x_i)^2
  $$
  이 **최소가 되도록** 𝛼와 𝛽를 추정하는 방법

![image-20221028225024299](C:/Users/yes47/AppData/Roaming/Typora/typora-user-images/image-20221028225024299.png)



- 𝛼에 대한 최소제곱 추정량
  $$
  \hat{\alpha} = \bar{y}-\hat{\beta}\bar{x}
  $$

- 𝛽에 대한 최소제곱 추정량
  $$
  \hat{\beta} = \frac{\sum_{i=1}^nx_i(y_i-\bar{y})}{\sum_{i=1}^nx_i(x_i-\bar{x})}, 단,\ \bar{x}는\ x_i의\ 평균,\ \bar{y}는\ y_i의\ 평균
  $$

- 𝑦i의 추정치
  $$
  \hat{y_i} = \hat{\alpha} + \hat{\beta}x_i, i=1,2,...,n
  $$

- 잔차(residual)
  $$
  e_i = y_i - \hat{y_i} = y_i - \hat{\alpha} - \hat{\beta}x_i\ , i=1,2,...,n
  $$
  

### 예제

![image-20221028225658054](C:/Users/yes47/AppData/Roaming/Typora/typora-user-images/image-20221028225658054.png)



- 회귀계수 추정
  $$
  \bar{x} = 1.54,\ \bar{y} = 42.98
  \\
  \hat{\beta} = \frac{\sum_{i=1}^nx_i(y_i-\bar{y})}{\sum_{i=1}^nx_i(x_i-\bar{x})} = 3/932
  \\
  \hat{\alpha} = \bar{y}-\hat{\beta}\bar{x} == -17.579
  $$

- 추정된 회귀선
  $$
  \hat{y} = -17.579 + 3.932x
  $$

- 결정계수
  $$
  R^2 = \frac{SSR}{SST} = 0.6511
  $$
  



## 단순선형회귀모형의 유의성 검정

### 모형의 유의성 t 검정

- 독립변수 x가 종속변수 Y를 설명하기에 유용한 변수인가에 대한 통계적 추론은 회귀계수 수 𝛽에 대한 검정을 통해 파악할 수 있음

- 가설

  H0 : 𝛽 = 0

  H1 : 𝛽  ≠ 0



- 검정통계량과 표본분포

  - 귀무가설 H0 이 사실일 때,

    <img src="C:/Users/yes47/AppData/Roaming/Typora/typora-user-images/image-20221028230243828.png" alt="image-20221028230243828" style="zoom:67%;" />

  - -> 독립변수 x가 종속변수 Y를 설명하기에 유용한 변수라고 해석할 수 있음



![image-20221028230357496](C:/Users/yes47/AppData/Roaming/Typora/typora-user-images/image-20221028230357496.png)



## 단순선형회귀모형의 적합도

### Y의 변동성 분해

![image-20221028230430159](C:/Users/yes47/AppData/Roaming/Typora/typora-user-images/image-20221028230430159.png)

- 결정계수 R^2
  $$
  R^2 = \frac{SSR}{SST} = 1-\frac{SSE}{SST}
  $$

  - SST = SSR+SSE 이므로 항상 0과 1사이의 값을 가짐 (0 <= R^2 <= 1)
  - y_i의 변동 가운데 추정된 회귀모형으로 통해 설명되는 변도으이 비중을 의미함
  - 0에 가까울수록 추정된 모형의 설명력이 떨어지는 것으로,
  - 1에 가까울수록 추정된 모형이 y_i 의 변동을 완벽하게 설명하는 것으로 해석할 수 있음
  - R^2는 두 변수 간의 상관계수 r의 제곱과 같음



## 문제



**Q1. 다음은 선형회귀모형에 관한 것이다. 올바른 설명은 무엇인가?**

1. 다중회귀모형에서는 종속변수가 여러 개다.

2. 단순회귀모형에서는 독립변수가 여러 개다.

3. 다중회귀모형에서의 종속변수는 범주형이다.

4. **단순회귀모형은 하나의 독립변수와 하나의 종속변수에 관한 모형이다.**



- 다중회귀모델은 독립변수가 여러 개, 종속변수가 하나이며, 회귀모델의 종속변수는 숫자형이어야 한다.



**Q2. 단순 선형회귀모형에서 오차항에 관한 가정으로 적절하지 않은 것은?**

1. 오차는 서로 독립임.

2. 오차는 정규분포를 따름.

3. **오차의 분산이 알려짐.**

4. 오차의 기대값은 0임.



- 오차는 **정규, 등분산, 독립 가정**이 필요하다. 오차의 분산은 일반적으로 알려져 있지 않은 것으로 가정된다.



**Q3. 두 숫자형 변수 X와 Y에 관한 다음 3개의 관찰값이 주어져 있다고 하자. 최소제곱법에 의해 단순선형회귀모형을 적합한다면, 기울기계수는 얼마로 추정되는가?**

![img](https://webcachecdn.multicampus.com/prd/plm/so/S000040000/S000046000/S000046350/S000046350_IM_91444.png?px-time=1666969708&px-hash=d2403cace278e1d6d77c34e196f6bb02)



- $$
  \hat{\beta} = \frac{\sum_{i=1}^nx_i(y_i-\bar{y})}{\sum_{i=1}^nx_i(x_i-\bar{x})}
  $$

- $$
  \bar{x} = 0 ,\ \bar{y}=0
  $$

- 

- 
  $$
  \hat{\beta} = \frac{(-2*-4)+(0*1)+(2*3)}{(-2*-2)+(0*0)+(2*2)} = 14/8 = 1.75
  $$



- X와 Y의 평균이 모두 0이므로, 최소제곱법에 의한 기울기 계수 베타의 추정치는 ((-2)*(-4)+0*1+2*3)/((-2)*(-2)+0*0+2*2)=14/8=1.75로 구해진다.

