## 문제 02

-----------

- Q1. EDA

  ```python
  # 여성으로 혈압이 High, Cholesterol이 Normal인 환자의 전체에 대비한 비율
  q1 = dat2[['Sex','BP','Cholesterol']].value_counts(normalize=True)
  
  q1.index
  # multi index
  
  q1[('F','HIGH','NORMAL')]
  ```

  

- Q2. 독립성 검정

  ```
  # 2. Age, Sex, BP, Cholesterol 및 Na_to_k 값이 Drug 타입에 영향을 미치는지 확인하기 위하여 아래와 같이 데이터를 변환하고 분석을 수행하시오.
  ```

  

  - Q2 -1. 데이터 변환

    ```python
    # 변수 생성(파생변수 생성)
    q2 = dat2.copy()
    
    # (1) Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은 ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오. 
    
    q2['Age_gr'] = np.where(q2['Age'] < 20, 10,
                     np.where(q2['Age'] < 30, 20,
                        np.where(q2['Age'] < 40, 30,
                            np.where(q2['Age'] < 50, 40,
                                np.where(q2['Age'] < 60, 50, 60)))))
    
    # (2) Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 초과는 ‘Lv4’로 변환하시오.
    
    q2['Na_K_gr'] = np.where(q2['Na_to_K'] <= 10, 'Lv1',
                        np.where(q2['Na_to_K'] <= 20, 'Lv2',
                            np.where(q2['Na_to_K'] <= 30, 'Lv3', 'Lv4')))
    
    # 결측치가 있다면
    # 결측치는 그대로 결측치로 둔 상태에서 나머지 값들을 변경
    q2['Na_K_gr'] = np.where(q2['Na_to_K'].isna(), q2['Na_to_K'],
                       np.where([q2['Na_to_K'] <= 10, 'Lv1',
                         np.where([q2['Na_to_K'] <= 20, 'Lv2',
                           np.where([q2['Na_to_K'] <= 30, 'Lv3', 'Lv4']))))
    ```

    > `np.where(조건)`  # 조건에 해당하는 행 번호가 리터
    >
    > `np.where(조건, 참인 경우 실행문, 거짓인 경우 실행문)`

  

  - Q2 -2. 카이스퀘어 검정

    ```python
    # - Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을 수행하시오.
    
    from scipy.stats import chi2_contingency # 카이스퀘어 검정
    
    # (1) 입력으로 들어갈 교차표 작성
    tab = pd.crosstab(index=q2['Sex'] , columns=q2['Drug'])
    
    # (2) 카이스퀘어 검정 수행
    chi2_contingency(tab)
    ```

    > 카이스퀘어 검정 해석
    >
    > ```
    > chi2_contingency(tab)
    > # 2.119248418109203,   # chi2
    > # 0.7138369773987128,   # p-value
    > #  4,  # 자유도(df)
    > #  array([[43.68, 11.04,  7.68,  7.68, 25.92],
    > #         [47.32, 11.96,  8.32,  8.32, 28.08]]))  # 기대빈도
    > ```

    ```python
    # 자동화
    var_list = ['Sex','BP','Cholesterol','Age_gr','Na_K_gr']
    
    q2_out = []
    for i in var_list:
        tab = pd.crosstab(index=q2[i] , columns=q2['Drug'])
        chi2_out = chi2_contingency(tab)
        q2_out.append([i, chi2_out[1]])
    
    q2_out = pd.DataFrame(q2_out, columns=['var','pvalue'])
    ```

    > H0 : 서로 독립이다
    >
    > H1 : 서로 독립이 아니다 (상관이 있다)
    >
    > 결론 : pvalue가 0.05보다 작은 변수가 연관성이 있는 변수

  - Q2 -3. 카이스퀘어 검정 후 연관성 있는 변수

    ```python
    q2_out2 = q2_out[q2_out.pvalue < 0.05]
    len(q2_out2)  # 4
    
    q2_out2.pvalue.max()  # 0.00070
    ```

  

- Q3. 의사결정나무를 이용한 분석 수행
  - 



- 의사결정나무





## 문제 03.

--------------------------------

- Q1. 평균으로부터 3 표준편차 밖의 경우를 이상치로 할 때 해당하는 데이터 개수

  ```python
  # 이상치 존재 파악
  
  q1 = dat3.copy()
  
  # 비율 변수 생성
  q1['forehead_ratio'] = q1['forehead_width_cm'] / q1['forehead_height_cm']
  
  
  # 비율 변수에 대한 평균, 표준편차 산출
  xbar = q1['forehead_ratio'].mean()
  std = q1['forehead_ratio'].std()
  
  UB = xbar + (3 * std)
  LB = xbar - (3 * std)
  
  # 비율 데이터와 범위 비교해서 이상치 유무 파악
  
  ((q1['forehead_ratio'] > UB) | (q1['forehead_ratio'] < LB)).sum()
  
  ```



- Q2. 이분산 통계 검정

  ```python
  #성별에 따라 forehead_ratio 평균에 차이가 있는지 적절한 통계 검정을 수행
  
  # (1) 성별에 따른 forehead_ratio
  q2_m = q1[q1.gender == 'Male']['forehead_ratio']
  q2_f = q1[q1.gender == 'Female']['forehead_ratio']
  
  from scipy.stats import ttest_ind
  
  # (2) 검정은 이분산을 가정하고 수행
  
  q2_ttest_out = ttest_ind(q2_m, q2_f, equal_var = False)
  # Ttest_indResult(statistic=2.9994984197511543, pvalue=0.0027186702390657176)
  q2_ttest_out
  
  # (3) 검정통계량의 추정치는 절대값을 취한 후 소수점 셋째 자리까지 반올림하여 기술
  round(abs(q2_ttest_out.statistic),3)
  q2_ttest_out.pvalue < 0.01  # 신뢰수준 99%에서 # True
  
  # 유의수준 0.01 하에서 유의수준 0.01보다 pvalue가 작으므로 귀무가설 기각
  ```



- Q3. 로지스틱 회귀분석

  

