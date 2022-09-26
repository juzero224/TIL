# ROUGE Score

- Recall-Oriented Understudy for Gisting Evaluation

- label(사람이 만든 요약문)과 summary(모델이 생성한 inference)을 비교해서 성능 계산

- ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S 등 다양한 지표가 있음

- 각각 지표별로 recall 및 precision을 둘 다 구하는 것이 도움이 됨

  -> F1 score 측정 가능

- Recall : label을 구성하는 단어 중 몇 개가 inference와 겹치는가?
  $$
  \frac{Number\ of\ overlapped\ words}{Total\ words\ in\ reference\ summary}
  $$
  

  - 우선적으로 필요한 정보들이 다 담겨있는지 체크

  

- Precision : inference를 구성하는 단어 중 몇 개각 label과 겹치는가?
  $$
  \frac{Number\ of\ overlapped\ words}{Total\ words\ in\ system\ summary}
  $$
  

  - 요약된 문장에 필요한 정보만을 얼마나 담고있는지를 체크



## 1. ROUGE-N

- N-gram 의 개수가 기준
- Rouge-1 은 reference와 model summary 간 겹치는 unigram 수를 보는 지표
- Rouge-2는 reference와 model summary 간 겹치는 bigram 수를 보는 지표
- Recall : output과 겹치는 N-gram 수 / label의 N-gram 수

- ![img](https://s1.md5.ltd/image/1349a0bbfa66a5fc3cc6cda5b58826ef.png)
  [ROUGE-1 SCORE(recall) example]
- precision : label과 겹치는 N-gram의 수 / output의 N-gram의수



### 2. ROUGE-L

- LCS(longest common sequence) between model output
- 말 그대로 common sequence 중에서 가장 긴 것을 매칭함
- n-gram과 달리 순서나 위치관계를 고려한 알고리즘
- Recall : LCS 길이 / label의 N-gram의수
- Precision : LCS 길이 / output의 N-gram의수



### 3. ROUGE-S

- Skip-bigram을 활용한 metric
- `Skip-gram Co-ocurrence`
- 두 개의 토큰을 한 쌍으로 묶어서(nC_2nC2) ROUGE Score를 계산
- 예를 들어, 'the brown fox' 는 (the,brown), (brown,fox), (the,fox)로 매핑되어 계산됨.
  ![img](https://s1.md5.ltd/image/ad954d1f266cabdb3a1ef40a4b8c2d71.png)
- 결과적으로 위 그림의 경우 _4C_24C2 와 _2C_22C2 개의 토큰 사이에서 비교하게 됨.(precision은 1/6이 됨).

```python
WINDOW_SIZE = 2

Ex. cat in the hat
=> skip-bigram = {"cat in","cat the","cat hat","in the","in hat","the hat"}
```





### 3-1. ROUGE-SU

- Unigram과 skip-bigram 모두 고려.
- 3번의 그림의 경우 output에 unigram인 the, brown, fox, jumps 를 추가
- (4+4C2)개, (2+2C2)개와 겹치는 개수를 비교하게 됨

### 4. Rouge score의 단점

- 동음이의어에 대해서 평가할 수 없음. 즉 같은 의미의 다른 단어가 많으면 성능을 낮게 측정함.
- Reference를 활용하여 average score를 내는 방법을 고려해 볼 수 있음
- 또는 동의어 dictionary를 구축하여 Rouge score를 계산하는 Rouge 2.0 방법도 있음

---



1. rouge library 설치

   ```python
   pip install rouge
   ```

2. `get.scores`

   ```python
   from rouge import Rouge
   
   model_out = ["he began by starting a five person war cabinet and included chamberlain as lord president of the council",
                "the siege lasted from 250 to 241 bc, the romans laid siege to lilybaeum",
                "the original ocean water was found in aquaculture"]
   
   reference = ["he began his premiership by forming a five-man war cabinet which included chamberlain as lord president of the council",
                "the siege of lilybaeum lasted from 250 to 241 bc, as the roman army laid siege to the carthaginian-held sicilian city of lilybaeum",
                "the original mission was for research into the uses of deep ocean water in ocean thermal energy conversion (otec) renewable energy production and in aquaculture"]
   rouge = Rouge()
   rouge.get_scores(model_out, reference, avg=True)
   
   >>>
   {   'rouge-1': {   'f': 0.6279006234427593,
                      'p': 0.8604497354497355,
                      'r': 0.5273531655225019},
       'rouge-2': {   'f': 0.3883256484545606,
                      'p': 0.5244559362206421,
                      'r': 0.32954545454545453},
       'rouge-l': {   'f': 0.6282785202429159,
                      'p': 0.8122895622895623,
                      'r': 0.5369305616983636}}
   ```

   