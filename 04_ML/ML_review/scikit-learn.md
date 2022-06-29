## 사이킷런

[사이킷런](https://scikit-learn.org/stable/)

- 기계학습 알고리즘
  - 지도학습 >> 목표변수 있음
    - 분류 (Y가 이산값)
      - KNN, Random Forest, Support Vector Machine, ANN
    - 추정(예측) (Y가 연속값)
      - Linear Regression, Regression Tree
  - 비지도학습 >> 목표변수 없음
    - 차원 축소 (Dimension Reduction)
      - PCA, MDS
    - 군집화 (Clustering)
      - Hierarchical clustering
    - 연관성 규칙 발견(Association Rule)



- 사이킷런 지도학습 (Supervised Machine Learning)
  - Model  << Training Data, Training Labels
    - `clf = RandomForestClassifier()`
    - `clf.fit(x_train, y_train)`
  - Prediction << Test Data
    - `y_pred = clf.predict(x_test)`
  - Evaluation << Test Labels
    - `clf.score(x_test, y_test)`



- 사이킷런 비지도 학습
  - Model << Traing Data
    - `pca = PCA()`
    - `pca.fit(x_Train)`
  - New View << Test Data
    - `x_new = pca.transform(x_test)`
  - 차원 축소, 클러스터링, 피처 추출
  - `fit()` : 입력 데이터의 형태에 맞춰 데이터를 변환하기 위한 사전 구조를 맞추는 작업
  - `transform()` : 입력 데이터의 차원 변환, 클러스터링, 피처 추출 등의 실제 작업

<br>

<br>

- 사이킷런 algorithm cheat-sheet

![scikit-learn cheet-sheet](scikit-learn-imgaes/Scikit-learn%20algorithm%20cheat-sheet.png)





사이킷런 주요 모듈

- 예제 데이터
  - `sklearn.datasets`
- 데이터 분리, 검증 & 파라미터 튜닝
  - `sklearn.model_selection`
- 피처 처리
  - `sklearn.preprocessing`
  - `sklearn.feature_selection`
    - 알고리즘에 큰 영향을 미치는 피처를 우선순위 대로 셀렉션 작업 수행
  - `sklearn.feature_extraction`
    - 텍스트 데이터나 이미지 데이터의 벡터화된 피쳐 추출
    - 텍스트 데이터 - Count Vectorizer, tf-idf Vectorizer
    - 텍스트 데이터 피처 추출 :  sklearn.feature_extraction.text
    - 이미지 데이터 피처 추출 : sklearn.feature_extraction.img
- 피처 처리 & 차원 축소
  - `sklearn.decomposition`
    - PCA, NMF, Truncated SVD
- 머신러닝 알고리즘
  - `sklearn.ensemble` : 앙상블 알고리즘
  - `sklearn.linear_model` : 선형 회귀, 릿지, 라쏘 및 로지스틱 회귀
  - `sklearn.naive_bayes` : 나이아베이즈 알고리즘
  - `sklearn.neighbors` : 최근접 이웃 알고리즘 K-NN
  - `sklearn.svm` : 서포트 벡터 머신 알고리즘
  - `sklearn.tree` : 의사 결정 트리 알고리즘
- 평가
  - `sklearn.metrics`
- 유틸리티
  - `sklearn.pipeline`



사이킷런 내장 예제 데이터 셋

- datasets.load_boston() : 회귀 용도, 미국 보스턴의 집 피처들과 가격에 대한 데이터 세트
- datasets.load_breast_cancer() : 분류 용도, 유방암 피처들과 악성/음성 레이블 데이터 세트
- datasets.load_diabetes() : 회귀 용도, 당뇨 데이터 세트
- datasets.load_digits() : 분류 용도, 0에서 9까지 숫자의 이미지 픽셀 데이터 세트
- datasets.load_iris() : 분류 용도, 붓꽃에 대한 피처를 가진 데이터 세트



- `fit()` : 사이킷런 ML 모델 ==학습== ==fit==
- `predict()` : 학습된 모델 ==예측== ==predict==



## 홀드 아웃(Hold Out)

- 데이터를 훈련 데이터와 테스트 데이터로 나눔
- `train_test_split()`
- Train/Test 비율 7:3 8:2 ...

```python
X_train, X_test, y_train, y_test = train_test_split(iris_data.data,
                                                   iris_data.target,
                                                   test_size = 0.3,  # 테스트 세트 크기를 얼마로 샘플링 할 것인가, default = 0.25 (25%)
                                                   #shuffle=True   # default = True, 데이터를 분리하기 전에 데이터를 미리 섞을지 결정
                                                   random_state=42)  # 재현율을 보장받기 위해서는 옵션 설정
```



## 교차 검증(K-Fold Cross Validation)

- 총 데이터 개수가 적은 데이터 셋에 대하여 정확도를 향상시킬 수 있음

1. K-Fold 교차 검증
   - K개의 데이터 폴드 세트를 만들어서 K번만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행하는 방법

2. Stratified K-Fold

   - 불균형(inbalanced) 분포도를 가진 레이블(결정 클래스) 데이터 집합을 위한 K-폴드 방식

   - 원본 데이터의 레이블 분포를 고려한 뒤 이 분포와 동일하게 학습과 검증 데이터 세트에 분배

3. `cross_val_score()` 함수 이용하여 교차검증
   - 폴드 세트 추출, 학습/예측, 평가를 한번에 수행
   - estimator
     - `classifier(분류)` 가 입력되면 `Stratified K 폴드`방식으로 학습/테스트 세트 분할
     - `regressor(회귀)` 가 입력되면 `K 폴드 방식`으로 분할

```python
cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=None,
               verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan,)
```

<br>

<br>

## 데이터 분석 실수

과대적합(Overfitting)

- 모델이 훈련 데이터에 너무 잘 맞지만 일반성이 떨어진다는 의미
- 훈련 데이터 이외의 다양한 변수에는 대응하기 힘들어짐
- 해결방법
- 1) 훈련 데이터를 더 많이 모음
  2) 정규화(Regularization) - 규제(제약 조건), 드롭-아웃 등 다양한 방법 이용
  3) 훈련 데이터 잡음을 줄임(오류 수정과 이상치 제거)



과소적합(Underfitting)

- 모델이 너무 단순해서 데이터의 내재된 구조를 학습하지 못함
- 해결방법
- 1. 파라미터가 더 많은 복잡한 모델 선택
  2. 모델의 제약 줄이기 (제약 하이퍼 파라미터 값 줄이기)
  3. 조기종료 시점(overfitting 되기 전 시점)까지 충분히 학습







보간법

[참고](https://iskim3068.tistory.com/35)