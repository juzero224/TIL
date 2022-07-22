## 문제 01

-----------

- Q1. 데이터 세트 내에 총 결측값의 개수

  ```python
  # 결측치가 포함된 '셀'의 수
  dat1.isnull().sum().sum()
  
  # 결측치가 포함된 '행'의 수
  dat1.isnull().any(axis=1).sum()
  ```

  > `any()` : 적어도 하나가 있다면, 열방향으로 체크 >> 행 방향으로 지정 해야 함

<br>

<br>

- Q2.  상관분석

  ```python
  # 세 가지 마케팅 채널의 예산과 매출액의 상관분석
  # 종속변수와 가장 강한 상관관계를 가지고 있는 독립변수의 상관계수
  
  q2 = dat1[['TV', 'Radio', 'Social_Media', 'Sales']].corr()
  q2_corr = q2['Sales'].drop('Sales').abs()
  
  q2_corr.max()  # 최댓값
  
  round(q2_corr.max(),4)
  
  # [참고]
  # [값과 인덱스]
  q2_corr.nlargest(1)
  
  # [최대 값이 있는 위치 번호]
  q2_corr.argmax()
  
  # [최댓값이 있는 인덱스명]
  q2_corr.idxmax()
  
  # [조건에 해당하는 인덱스명]
  q2_corr[q2_corr >= 0.6].index
  
  # 시각화
  import seaborn as sns
  
  sns.pairplot(dat1[['TV', 'Radio', 'Social_Media', 'Sales']])
  ```
  
  > `abs()` : 절대값
  >
  > ![image-20220714234953690](proDS-imgaes/image-20220714234953690.png)
  >
  > > radio와 social media는 선형이지만 관계가 너무 넓게 퍼져있다.
  > >
  > > Social media와 Sales는 선형 관계가 없음
  >
  > cov(x,y) : 공분산  
  > $$
  > E[(x_i-\mu_x)(y_i-\mu_y)]
  > $$
  >
  > $$
  > \sigma_x \sigma_y < cov(x,y) < \sigma_x \sigma_y
  > $$
  >
  > corr(x,y) : 상관계수  -1 < corr(x,y) < 1
  > $$
  > corr(x,y) = \sigma_x / \sigma_y
  > $$
  > 

<br>

<br>

- Q3. 회귀분석

  ```python
  # 매출액을 종속변수, TV, Radio, Social Media의 예산을 독립변수로 하여 회귀분석을 수행하였을 때, 세 개의 독립변수의 회귀계수를 큰 것에서부터 작은 것 순으로 기술
  
  # 결측치 제거
  q3 = dat1.dropna()
  
  var_list = ['TV', 'Radio', 'Social_Media']
  
  lm = LinearRegression(fit_intercept=True).fit(dat2[var_list], dat2['Sales'])
  # 독립변수가 하나만 들어가도 되지만 2차원이어야 함
  lm_TV = LinearRegression(fit_intercept=True).fit(dat2[['TV']], dat2['Sales'])
  
  # dir(lm)
  
  lm.intercept_ # 상수항 
  lm.coef_      # 회귀계수 
  
  q3_weight = pd.Series(lm.coef_, index=var_list)
  
  np.trunc(q3_weight.sort_values(ascending=False) * 1000) / 1000
  ```

  > 1. 결측치가 없어야 함
  >
  > 2. 입력될 때는 2차구조로 들어가야 한다(2D가 맞는지 확인)
  >
  >    `[[ D , D], [D , D]]`

<br>

<br>

- Q3 -2. 모수적 방법으로 해석 - 이상치 제외 안했을 경우

  `ols(식, 데이터셋).fit()`

  - 식 : `Y ~ X1 + C(X2) + X3 -1`
  - `-1` : 상수항 미포함
  - `C()` : 범주형 선언 -> 더미 변수로 자동 변환 생성
  - 변수들 차이를 나타내는 차이값을 가중치로 구하도록 하는 기법을 더미변수
  - 결측치는 자동 제외

  ```python
  form1 = 'Sales~' + '+'.join(var_list)
  lm2 = ols(form1, dat1).fit()
  
  # dir(lm2)
  
  lm2.summary()
  ```

  > 반드시 2차 식으로 넣을 필요 없다

  ```
  lm2.summary()
  #   OLS Regression Results                            
  # ==============================================================================
  # Dep. Variable:                  Sales   R-squared:                       0.999
  # Model:                            OLS   Adj. R-squared:                  0.999
  # Method:                 Least Squares   F-statistic:                 1.505e+06
  # Date:                Thu, 14 Jul 2022   Prob (F-statistic):               0.00
  # Time:                        13:47:05   Log-Likelihood:                -11366.
  # No. Observations:                4546   AIC:                         2.274e+04
  # Df Residuals:                    4542   BIC:                         2.277e+04
  # Df Model:                           3                                         
  # Covariance Type:            nonrobust                                         
  # ================================================================================
  #                    coef    std err          t      P>|t|      [0.025      0.975]
  # --------------------------------------------------------------------------------
  # Intercept       -0.1340      0.103     -1.303      0.193      -0.336       0.068
  # TV               3.5626      0.003   1051.118      0.000       3.556       3.569
  # Radio           -0.0040      0.010     -0.406      0.685      -0.023       0.015
  # Social_Media     0.0050      0.025      0.199      0.842      -0.044       0.054
  # ==============================================================================
  # Omnibus:                        0.056   Durbin-Watson:                   1.998
  # Prob(Omnibus):                  0.972   Jarque-Bera (JB):                0.034
  # Skew:                          -0.001   Prob(JB):                        0.983
  # Kurtosis:                       3.013   Cond. No.                         149.
  # ==============================================================================
  ```
  > R-squared : 결정계수, 전체 데이터 중 해당 회귀모델이 설명할 수 있는 데이터의 비율 회귀식의 설명력을 나타낸다. 성능을 나타냄 ( 1에 가까울수록 성능이 좋음)
  > Adj.R-Squared : 모델에 도움이 되는 데이터에 따라 조정된 결정계수
  >F-statistic : F 통계량으로 도출된 회귀식이 적절한지 볼 수 있음. 0과 가까울 수록 적절
  >Prob(F-statistic) : 회귀식이 유의미한지 판단 (0.05 이하일 경우 변수끼리 매우 관련있다고 판단)
  > AIC : 표본의 개수와 모델의 복잡성을 기반으로 한 모델을 평가, 수치가 낮을 수록 좋음
  >BIC : AIC와 유사하나 패널티를 부여하여 AIC보다 모델 평가 성능이 더 좋고, 수치가 낮을 수록 좋음
  <br>

  > - y - y_hat = 잔차가 최소되도록 least squares 방법을 사용한 것
  >
  > - 왜도(skew) : 정규분포의 꼬리가 얼마나 긴지
  > - 첨도(kurtosis) : 평균 주변에 얼마만큼 모여 있느냐
  > - Durbin-Watson: 자기 상관
  >   - Durbin-Watson 통계량은 0 < d < 4의 값을 가집니다.
  >   - d ~ 0 : 잔차끼리 양의 상관관계를 가진다.
  >   - d ~ 2: 잔차끼리 상관관계를 가지지 않는다.
  >   - d ~ 4: 잔차끼리 음의 상관관계를 가진다.

<br>

<br>

- Q3 -3. 모수적 방법으로 해석 - 이상치 제외 했을 경우

  ```python
  lm2.params
  
  sel_var_list = lm2.pvalues[lm2.pvalues < 0.05].index
  lm2.params[lm2.pvalues < 0.05]
  
  # 이상치 확인
  outlier = lm2.outlier_test()
  # < 0.05 : 이상치 / > 0.05 : 이상치 아님
  
  # 이상치 제외 데이터 획득
  dat2[outlier['bonf(p)'] >= 0.05] 
  
  form2 = 'Sales~' + '+'.join(dat1.columns.drop('Sales'))
  ```

<br>

<br>

- Q3 -4. 모수적 방법으로 해석 - 범주형 변수 삽입 시

  ```python
  form2 = 'Sales~' + '+'.join(dat1.columns.drop('Sales'))
  
  lm3 = ols(form2, dat2).fit()
  # lm3.summary()
  lm3.predict(dat2.drop(columns=['Sales']))
  # 변주형 변수도 더미로 바꿔서 계산됨
  
  anova1 = ols('Sales~Influencer', dat2).fit()
  
  from statsmodels.stats.anova import anova_lm
  anova1_lm(anova1)
  
  # 다중 비교, 사후 분석
  from statsmodels.stats.multicomp import pairwise_tukeyhsd
  print(pairwise_tuckeyhsd(dat2['Sales'], dat2['Influencer']))
  ```

  > 해석

  ```
  anova_lm(anova1)
  # 분산분석표
  #                 df        sum_sq      mean_sq         F    PR(>F)
  # Influencer     3.0  2.081064e+04  6936.879515  0.801596  0.492813
  # Residual    4542.0  3.930570e+07  8653.830120       NaN       NaN
  ```





---------------------------

## 회귀분석

- 회귀분석
  - 변수들간의 함수적인 관련성을 규명하기 위해 수학적 모형을 가정하고, 이 모형을 측정된 변수들의 자료로 부터 추정하는 통계적 분석 방법
  - 종속 변수 : 다른 변수의 영향을 **받는** 변수
  - 독립 변수 : 다른 변수에 영향을 **주는** 변수

<br>

- 선형회귀

  - *ε* : 오차를 나타내는 확률 변수

  - *β*0 , *β*1, ... , *β*k : 회귀계수 (regression coefficients)

  - 

  - $$
    Y_i = \alpha + \beta x_i + \varepsilon_i,\ \varepsilon_i \sim N(0, \sigma^2)
    $$

- 

<br>

- 최소제곱법 (method of least squares)

  - 오차제곱합(error sum of squares)을 최소로 하도록 추정하는 방법

  - $$
    SS = \sum_{i=1}^n \varepsilon_i^2 = \sum_{i=1}^n (Y_i - \alpha - \beta x_i)^2
    $$

  - $$
    S_{xy} = \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) = \sum_{i=1}^n x_i,y_i - \frac{( \sum_{i=1}^n x_i) (\sum_{i=1}^n y_i)}{n}
    $$

<br>

- 최소 제곱 회귀직선

  - $$
    \hat{y} = \hat{\alpha} + \hat{\beta}x
    $$

  - $$
    \hat{y} = \bar{y} + \hat{\beta}(x-\bar{x})
    $$

  - residual(잔차) e :

  - $$
    e_i = Y_i - \hat{Y}_i
    $$

<br>

-  불편 추정량

  - 잔차 제곱합 (residual sum of squares)

  - $$
    SSE = \sum_{i=1}^n (Y_i-\hat{Y}_i)^2
    $$

  - 분산 sigma^2의 불편 추정량 (잔차평균제곱) (residual mean square)

  - $$
    MSE = SSE/(n-2) = \sum_{i=1}^n (Y_i - \hat{\alpha} - \hat{\beta}x_i)^2 / (n-2)
    $$

<br>

- 

- 최소제곱추정량 *β*^(hat)

  - 단순회귀 모형에서 회귀직선의 기울기 *β* 추정량

    - $$
      \hat{\beta} = \frac{S_{xy}}{S_{xx}} = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i-\bar{Y})}{\sum_{i=1}^n (x_i - \bar{x})^2}
      $$

  - 

  - 1. *β*^는 정규분포를 따름

    - $$
      \hat{\beta} \sim N(\beta, \frac{\sigma^2}{S_{xx}})
      $$

  - 2. 잔차평균제곱 MSE = SSE/(n-2)는 *σ*^2의 불편 추정량

    - $$
      E(MSE) = \sigma^2
      $$

  - 3. Var(*β*^) 의 불편추정량은 MSE/S_(xx)

  - 4. t분포를 따르고 자유도는 n-2
       $$
       \frac{\hat{\beta} - \beta}{\sqrt{\frac{MSE}{S_{xx}}}} \sim t(n-2)
       $$
       

<br>



1. 선형성 O = 회귀식이 존재한다

2. 독립성 - *ε_i*  (잔차(e)에 의해 ε 추정 가능) 

   -> 가정이 만족됐는지 잔차 분석을 통해 확인 필요

3. 정규성 - ε_i (잔차, 오차) 정규분포를 따른다 가정

4. 등분산성 



```
H0 : β1 = β2 = β3 = 0
H1 : 적어도 i 하나에 대해서 βi != 0
-> 회귀식 존재 유무
```









- SST(총 제곱합) = SSE(처리제곱합) + SSR(잔차제곱합)

$$
\sum(y_i-\bar{y})^2 
= \sum(y_i - \hat{y_i} + \hat{y_i} - \bar{y})^2
\\ = \sum(y_i-\hat{y_i})^2 + \sum(\hat{y_i}+\bar{y})^2
\\ SST = SSE + SSR
$$
<br>

$$
\frac{SST}{n-1} = \frac{SSE}{n-1-k} + \frac{SSR}{k}\\
                = MSE + MSR
$$
<br>
$$
R^2 = \frac{SSR}{SST} = 1-\frac{SSE}{SST}
$$
<br>
- 수정된 R스퀘어

$$
adj-R^2 = \frac{SSR/df}{SST/df}
$$


$$
F = \frac{MSR}{MSE} = \frac{SSR/k}{SSE/n-k-1}
$$




```
      |   제곱합  자유도  평균제곱	  F     	F(alpha)
---------------------------------------------------------------------
회귀   |   SSR    k	  MSR		MSR/MSE		F(k, n-k-1, ; alpha)   
잔차   |   SSE 	  n-k-1	  MSE
---------------------------------------------------------------------
계    |	  SST     n-1  
```



