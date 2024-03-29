# 14_규제가 있는 선형회귀모델 (Ridge, Lasso, Elastic Net)







## 문제

**Q1. 선형회귀모형에서 모형의 파라미터에 대한 제약을 가하는 규제(regularization)가 필요한 이유는 무엇인가?**

1. 모형의 편향을 줄이기 위해서 

2. **모형의 과적합을 방지하기 위해서**

3. 훈련자료가 부족해서

4. 모형의 변동성을 키우기 위해서



- 선형회귀모형에 파라미터에 대한 규제를 가하는 경우 **편향이 늘어나지만 변동성을 줄여줘 과적합을 방지할 수 있다**.



**Q2. 선형회귀모델에 대한 규제 방법에 대한 다음의 설명 중 잘못된 것은?**

1. 릿지회귀는 모델 파라미터들의 제곱합이 일정 수준을 넘지 않도록 제한한다.

2. 라쏘회귀는 모델 파라미터들의 절대값의 합이 일정 수준을 넘지 않도록 제한한다. 

3. **릿지회귀를 적용하면 일부 파라미터 추정치가 0이 되어, 변수선택의 효과가 있다.**

4. 엘라스틱넷은 릿지회귀와 라쏘회귀에서의 제약이 모두 반영된다.



- 일부 파라미터를 0으로 추정하여 변수선택의 효과가 있는 규제방식은 라쏘회귀이다.





**Q3. 규제가 있는 선형회귀모형(릿지, 라쏘, 엘라스틱넷 등)은 모형의 복잡도를 증가시켜서 예측력을 높이고자 하는 방법이다. (O, X)**



- **X**

- 규제는 모형의 복잡도를 낮춰서 과대적합을 줄이기 위한 기법이다.