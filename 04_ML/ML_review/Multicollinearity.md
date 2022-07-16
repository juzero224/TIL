# 다중회귀분석
## 다중 공선성
- 독립변수가 여러 개일 경우 그 변수들 끼리 상관관계가 높을 경우
- 다중 공선성 판단
  - 독립변수들끼리의 상관계수가 90% 이상
  - VIF(Variance Inflation Factor : 분산 확대 인자)
$$
VIF=1/(1-R^2)
$$
  - 공차한계 (Tolerance)를 통해서도 다중공선성을 판단
    - 공차한계 : 어떤 독립변수가 다른 독립변수들에 의해서 설명되지 않는 부분을 의미
$$
변수 i의 공차한계 = 1-R_i^2
$$
  - 결정계수가 클수록 -> 공차한계 값이 작아짐
  - 공차한계 값이 작을수록 -> 그 독립변수가 다른 독립변수들에 의해 설명되는 정도가 큼
  - = 다중공선성이 높다
$$
결정계수 \uparrow = 공차한계 \downarrow = 설명력 \uparrow = 다중공선성 \uparrow
$$
- VIF가 10이상이거나, 공차한계가 0.10이하일 경우 -> 공선성이 존재
<br>
- 다중공선성 의심 상황
  - Data 수에 비해 과다한 독립변수를 사용했을 때
  - 독립변수들의 상관계수가 크게 나타날 떄
  - 한 독립변수를 회귀모형에 추가하거나 제거하는 것이 회귀계수의 크기나 부호에 큰 변화를 줄 때 회귀계수의 크기나 부호에 큰 변화를 줄 때
  - 중요하다고 생각되어지는 독립변수에 대한 P값이 크게 나타나 통계적 차이가 없을 때
  - (회귀계수의 부호가 과거의 경험이나 이론적인 면에서 기대되는 부호와 정반대일 때)
<br>
- 다중 공선성 발생 -> 회귀모형의 적합성이 떨어짐 -> 다른 중요한 독립변수가 모형에서 제거될 가능성 높음
- 결정계수의 값이 과대하게 나타나거나 설명력을 좋은데 예측력이 떨어질 수 있음

- 공선성을 낮추는 방법 = 상관관계가 높은 독립변수 제거
- VIF가 10보다 큰 값 제거

<br>
```python
# 다중 공선성
def get_vif(formula, df):
    from patsy import dmatrices
    y, X = dmatrices(formula, df, return_type = "dataframe")
    vif = pd.DataFrame()
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif['VIF Factor'] = [variance_inflation_factor(X.values, i)
                        for i in range(X.shape[1])]
    vif['features'] = X.columns
    vif.sort_values(by='VIF Factor', ascending = False, inplace = True)
    return vif
```
```python
formula = "item_cnt_month~" + "+".join(validation_set.drop(['item_cnt_month'], axis=1).columns) + "-1"
get_vif(formula, validation_set)
```
<br>

- 상관계수와 VIF를 사용하여 독립 변수를 선택


```python
# 독립 변수만 뽑아서 다시 vif 검증 (ex)
def get_model2(seed):
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=seed)
    model = sm.OLS.from_formula("TOTEMP ~ scale(GNP) + scale(ARMED) + scale(UNEMP)", data=df_train)
    return df_train, df_test, model.fit()


df_train, df_test, result2 = get_model2(3)
print(result2.summary())
```
```python
test2 = []
for i in range(10):
    df_train, df_test, result = get_model2(i)
    test2.append(calc_r2(df_test, result))

test2
```
<br>

- 다중공선성 제거한 경우 학습 성능과 검증 성능간의 차이가 줄어듬

```python
# 다중공선성 제거 전과 후 비교 (ex)
plt.subplot(121)
plt.plot(test1, 'ro', label="검증 성능")
plt.hlines(result1.rsquared, 0, 9, label="학습 성능")
plt.legend()
plt.xlabel("시드값")
plt.ylabel("성능(결정계수)")
plt.title("다중공선성 제거 전")
plt.ylim(0.5, 1.2)

plt.subplot(122)
plt.plot(test2, 'ro', label="검증 성능")
plt.hlines(result2.rsquared, 0, 9, label="학습 성능")
plt.legend()
plt.xlabel("시드값")
plt.ylabel("성능(결정계수)")
plt.title("다중공선성 제거 후")
plt.ylim(0.5, 1.2)

plt.suptitle("다중공선성 제거 전과 제거 후의 성능 비교", y=1.04)
plt.tight_layout()
plt.show()
```