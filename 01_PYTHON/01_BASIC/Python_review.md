# :notebook_with_decorative_cover: 01_Python - 1. Python_intro



## 01장 파이썬 첫걸음

### 1. 출력하기

- 기본 출력 : `print(value1, value2, ...)`

- 구분자 지정

  - `print(value1, value2, ..., sep='구분자')`

- 끝문자 지정

  - `print(value,1 value2, ..., end='끝문자')`

- 여러 줄 한번에 출력하기

  - 홑따옴표 세개(''') 혹은 쌍따옴표 세개(""")로 묶는다

  - ```python
    print('''
    자세히 보아야 예쁘다.
    오래 보아야 사랑스럽다.
    너도 그렇다.
    ''')
    ```

- 주석달기
  * 프로그램에 대한 설명을 적을 때, 코드의 실행을 잠시 막아둘 때 주석을 사용한다.
  * 파이썬에서는 샵('#') 을 주석 기호로 사용한다.
- 들여쓰기
  * 파이썬에서는 들여쓰기 자체가 문법입니다.
  * 파이썬 들여쓰기 방법은 공백2칸, 공백4칸 등 여러가지 방법이 있지만 보통 공백 4칸을 사용합니다.
- 오류해석하기
  - 파이썬에서는 대소문자를 구분한다
  - 따옴표의 짝이 맞아야 한다
  - 괄호의 짝이 맞아야한다

<br>

<br>

### 2. 기본자료형 다루기

#### 2.1 변수 만들기

- 변수 생성
  - **변수** : 값을 저장하기 위한 객체(object)
  - 변수 명명 규칙
    - 대소문자 구분, 숫자 시작 불가(숫자 포함 가능), 특수기호(!@#) 삽입불가(_dict:_사용가능)
    - 예악어 불가 (함수명, 함수 내 인자명, 패키지 이름: if, for, while)

- (_dict:_사용가능)

- 예악어 (함수명, 함수 내 인자명, 패키지 이름: if, for, while)변수에 값 할당하기

  - 변수명 = 변수값
  - 변수에 값이 할당될 때 변수값의 자료형에 따라 변수의 자료형이 결정된다.

- 변수 값, 자료형 출력하기

  - 변수에 접근할 때는 변수명을 사용
  - 변수값 출력 : `print(변수명)`
  - 변수자료형 출력 : `print(type(변수명))`

- 쌍따옴표나 홑따옴표가 포함되어 있는 문자열 사용하기

  - '낮말'은 새가 듣고 '밤말'은 쥐가 듣는다.

  - ```python
    a = "'낮말'은 새가 듣고 '밤말'은 쥐가 듣는다."
    print(a)
    ```

<br>

<br>

#### 2.2 키보드로 변수에 값 입력받기

- `input()`

  - `input()`으로 입력받은 값은 무조건 `str`

- 자료형 변환하기

  - `int(value)` : int형으로 변환

  - `str(value)` : str형으로 변환

  - `float(value)` : float형으로 변환

  - input으로 입력받은 값은 숫자형으로 변환하여 사용하기

    ```python
    a = int(input('나이:'))
    ```

<br>

<br>

<br>

## 02장 파이썬 프로그래밍의 기초, 자료형

### 1. 산술연산

- **숫자형**

  - 숫자형(Number)이란 숫자 형태로 이루어진 자료형

    | 항목   | 사용 예                 |
    | ------ | ----------------------- |
    | 정수   | 123, -345, 0            |
    | 실수   | 123.45, -1234.5, 3.4e10 |
    | 8진수  | 0o34, 0o25              |
    | 16진수 | 0x2A, 0xFF              |

```python
print('a+b=',a+b) # 더하기
print('a-b=',a-b) # 빼기
print('a*b=',a*b) # 곱하기
print('a/b=',a/b) # 나누기
print('a//b=',a//b) # 몫
print('a%b=',a%b)   # 나머지
print('a**b=',a**b) # 제곱
```

<Br>

<br>

### 2. 문자열 연산

#### 2-1.문자열이란?

- **문자열**(String)이란?

  -  문자, 단어 등으로 구성된 문자들의 집합

- 이스케이프 코드

  | `\n`   | 문자열 안에서 줄을 바꿀 때 사용                         |
  | ------ | ------------------------------------------------------- |
  | `\t`   | 문자열 사이에 탭 간격을 줄 때 사용                      |
  | `\\`   | 문자 `\`를 그대로 표현할 때 사용                        |
  | `\'`   | 작은따옴표(`'`)를 그대로 표현할 때 사용                 |
  | `\"`   | 큰따옴표(`"`)를 그대로 표현할 때 사용                   |
  | `\r`   | 캐리지 리턴(줄 바꿈 문자, 현재 커서를 가장 앞으로 이동) |
  | `\f`   | 폼 피드(줄 바꿈 문자, 현재 커서를 다음 줄로 이동)       |
  | `\a`   | 벨 소리(출력할 때 PC 스피커에서 '삑' 소리가 난다)       |
  | `\b`   | 백 스페이스                                             |
  | `\000` | 널 문자                                                 |

<br>

<br>

#### 2-2 문자열 연산

- 문자열 더하기(문자열 연결하기)

  ```python
  # 문자열 더하기(문자열 연결하기)
  a = 'good'
  b = 'morning'
  a+b
  ```

  > ```
  > 'goodmorning'
  > ```

- 문자열 곱하기(문자열 반복하기)

  ```python
  # 문자열 곱하기(문자열 반복하기) 
  s = 'ha'
  s * 3
  ```

  > ```
  > 'hahaha'
  > ```

- 문자열과 숫자형 더하기

  ```python
  # 문자열과 숫자형 더하기
  english = 80
  result = '영어점수: '+ str(english)
  print(result)
  ```

  > ```
  > 영어점수: 80
  > ```

- 문자열 길이 구하기

  ```python
  a = "Life is too short"
  len(a)
  ```

<br>

<br>

#### 2-3. 문자열 인덱싱과 슬라이싱

- 한 문자씩 건너뛰어 출력하기

  ```python
  a = 'Hello World'
  print(a[::2])
  ```

- 역순으로 출력하기

  - 간격을 마이너스(-)로 하면 역순으로 출력된다.

  ```python
  a = 'Hello World'
  print(a[::-1])
  ```

<br>

<br>

#### 2-4. 문자열 포매팅

| 코드 | 설명                      |
| ---- | ------------------------- |
| %s   | 문자열(String)            |
| %c   | 문자 1개(character)       |
| %d   | 정수(Integer)             |
| %f   | 부동소수(floating-point)  |
| %o   | 8진수                     |
| %x   | 16진수                    |
| %%   | Literal % (문자 `%` 자체) |

```python
print('I eat %d apples' % 3)  # %d : 문자열(숫자) 포맷 코드
print('I have %s apples' % five) # %s : 문자열(문자) 포맷 코드
print('rate is %f' % 3.234)  # float
print('rate is %s' % 3.234)  # float -> 문자열
print('Error is %d%%.' % 98)  # %d뒤 % >> %d%%
```

<br>

- 정렬과 공백

  ```python
  print('%10s' % 'hi')  # %10s : 10개 공간 값 오른쪽 정렬
  print('%-10sjane.'% 'hi')  # 왼쪽 정렬
  ```

  > ```
  >         hi
  > hi        jane.
  > ```

- 소수점 표현

  ```python
  print('%0.4f' % 3.42134234)
  print('%10.4f' % 3.42134234)
  ```

  > ```
  > 3.4213
  >     3.4213
  > ```

- format 함수 사용한 포매팅

  ```python
  print('I eat {0} apples'.format(3))
  print('i eat {0} apples'.format(number))
  ```

  ```python
  print('{0:<10}'.format('hi'))  # :<10 왼쪽 정렬, 총 자릿수 10
  print('{0:>10}'.format('hi'))  # :>10 오른쪽 정렬, 총 자릿수 10
  print('{0:^10}'.format('hi'))  # :^10 가운데 정렬, 총 자릿수 10
  ```

  - 공백 채우기

  ```python
  # 공백 채우기
  print('{0:=^10}'.format('hi'))
  print('{0:!<10}'.format('hi'))
  ```

  - 소수점 표현

  ```python
  # 소수점 표현
  y = 3.42134234
  print('{0:0.4f}'.format(y))
  print('{0:10.4f}'.format(y))
  print('{{and}}'.format())
  ```

<br>

<br>

#### 2-5 f스트링으로 출력하기

- 방법1

  ```python
  age = int(input('나이:'))
  print('나이는',age,'살이시군요. 내년이면',age+1,'살이 되시겠네요')
  ```

- 방법2

  ```python
  print(f'나이는 {age}살이시군요. 내년이면 {age+1}살이 되시겠네요')
  ```

<br>

연습문제

* 화씨 온도를 입력받아 섭씨 온도로 변환하는 프로그램을 작성해보세요.  
  $$
  C = (F - 32) * \frac{5}{9}
  $$

  ```python
  # 화씨온도 입력
  f = float(input('화씨온도:'))
  
  # 섭씨온도로 계산
  c = (f - 32) * (5/9)
  
  # 결과 출력
  print(f'화씨온도:{f} --> 섭씨온도:{round(c,2)}')
  ```

  >```
  >화씨온도:80
  >화씨온도:80.0 --> 섭씨온도:26.67
  >```

<br>

<br>

#### 2-6. 문자열 관련 함수들

- 문자 개수 세기 : `count()`

  ```python
  a = 'hobby'
  a.count('b')
  # >> 2
  ```

- 위치 알려주기1 : `find()`

  ```python
  a = 'Python is the best choice'
  print(a.find('b'))
  print(a.find('k'))   # >> -1 : 찾는 문자나 문자열이 없을 때 -1 반환
  # >> 14
  # >> -1
  ```

- 위치 알려주기2 : `index()`

  ```python
  a = 'Life is too short'
  print(a.index('t'))
  # print(a.index('k'))
  # ValueError: substring not found
  # find와 차이점. 문자열이 없으면 오류가 발생
  ```

- 문자열 삽입 : `join()`

  ```python
  # 문자열 삽입(join)
  print(','.join('abcd') )
  # >> a,b,c,d
  ```

- 소문자를 대문자로 바꾸기 : `upper()`

- 대문자를 소문자로 바꾸기 : `lower()`

- 왼쪽 공백 지우기 : `lstrip()`

- 오른쪽 공백 지우기 : `rstrip()`

- 양쪽 공백 지우기 : `strip()`

- 문자열 바꾸기 : `replace()`

  ```python
  a = '나는 초코우유 좋아. 초코우유 최고'
  
  # '초코'-->'딸기'로 교체
  print(a.replace('초코','딸기'))
  ```

- 문자열 나누기 : `split()`

  ```python
  phone = '010-123-4567'
  phone.split('-')
  # >> ['010', '123', '4567']
  ```

<br>

연습문제

- 사용자의 영문 이름을 입력받아 성과 이름 순서를 바꾸어서 출력하는 프로그램을 작성하세요. 성과 이름은 공백으로 구분합니다.

  ```python
  # 사용자의 영문이름 입력받기 (성과 이름은 공백으로 구분)
  full_name = input('영문이름(성과 이름은 공백으로 구분하세요 : ')
  # 공백의 위치 찾기
  space = full_name.find(' ')
  print(space)
  # 성, 이름을 슬라이싱하여 각각 변수에 담기
  first_name = full_name[:space]
  last_name = full_name[space+1:]
  
  print(first_name)
  print(last_name)
  # 성, 이름의 순서를 바꾸어 출력하기
  print(last_name, first_name)
  ```

  > ```
  > 영문이름(성과 이름은 공백으로 구분하세요 : Juyeong Park
  > 7
  > Juyeong
  > Park
  > Park Juyeong
  > ```

<br>

<br>

### 3. 리스트

#### 3-1. 리스트 만들기

- 빈 리스트 만들기 
  1. `[]`를 이용 : `l1 = []`
  2. `list()` 함수 이용 : `l2 = list()`
- 초기값이 있는 리스트
  1. `[]`이용 : `l3 = [1,3,5,7,9]`
  2. `list()`함수 이용 : `l4 = list(range(1,100,2))`

<br>

<br>

#### 3-2 리스트 인덱싱과 슬라이싱

```python
a = [1, 2, ['a', 'b', ['Life', 'is']]]
a[2][2][0]
# 'Life'
```

```python
a = [1,2,3,['a','b','c'],4,5]
print(a[2:5])
print(a[3][:2])
# >> [3, ['a', 'b', 'c'], 4]
# >> ['a', 'b']
```

<br>

<br>

#### 3-3 리스트 연산하기

- 리스트 더하기

  ```python
  # 리스트 더하기
  a = [1,2,3]
  b = [4,5,6]
  a+b
  ```

- 리스트 반복하기

  ```python
  # 리스트 반복하기
  a = [1,2,3]
  a * 3
  ```

- 리스트 길이 구하기 : `len()`

  ```python
  # 리스트 길이 구하기
  a = [1,2,3]
  len(a)
  ```

<br>

<br>

#### 3-4 리스트 수정과 삭제

- 리스트 수정

  ```python
  # 리스트 값 수정
  a = [1,2,3]
  a[2] = 4
  a
  ```

<br>

- 리스트 요소 삭제 `del()`

  ```python
  # del 함수 : 리스트 요소 삭제
  a = [1,2,3,4,5]
  del a[1] # >> [1,3,4,5]
  del a[2:] # >> [1,3]
  a
  # >> [1, 3]
  ```

<br>

<br>

#### 3-5 리스트 관련 함수

- 리스트에 요소 추가 : `append()`

- 리스트 정렬 : `sort()`

- 리스트 뒤집기 : `reverse()`

- 위치 반환 : `index()`

- 리스트에 요소 삽입 : `insert()`

- 리스트 요소 제거 : `remove()`

- 리스트 요소 끄집어내기 : `pop()`

- 리스트에 포함된 요소 x의 개수 세기 : `count()`

- 리스트 확장 : `extend()`

- 리스트에 값 존재 여부 확인 : `in` `not in`

  ```python
  l = [1,2,3,4]
  print(5 in l)
  print(5 not in l)
  # >> False
  # >> True
  ```

<br>

<br>

### 4. 튜플

튜플(tuple)은 몇 가지 점을 제외하곤 리스트와 거의 비슷하며 리스트와 다른 점은 다음과 같다.

- 리스트는 [ ]으로 둘러싸지만 튜플은 ( )으로 둘러싼다.
- 리스트는 그 값의 생성, 삭제, 수정이 가능하지만 튜플은 그 값을 바꿀 수 없다.

 튜플과 리스트의 가장 큰 차이는 값을 변화시킬 수 있는가 여부이다. 즉 리스트의 항목 값은 변화가 가능하고 튜플의 항목 값은 변화가 불가능하다. 

<br>

- 빈 튜플 만들기 
  1. `()`를 이용 : `t1 = ()`
  2. `tuple()` 함수 이용 : `t2 = tuple()`
- 초기값이 있는 튜플
  1. `()`이용 : `t3 = (1,3,5,7,9)`
  2. `tuple()`함수 이용 : `t4 = tuple(range(1,100,2))`

- 튜플을 만들 때 괄호()를 생략할 수 있다.

  `t5 = 1,3,5,7,9`
  `t5, type(t5)`

  > ```
  > ((1, 3, 5, 7, 9), tuple)
  > ```

<br>

<br>

### 5. 딕셔너리

#### 5-1 딕셔너리란?

- 대응 관계를 나타내는 자료형
- 리스트나 튜플처럼 순차적으로(sequential) 해당 요솟값을 구하지 않고 Key를 통해 Value를 얻는다.



#### 5-2. 딕셔너리 만들기

- `딕셔너리명 = {키1:값1, 키2:값2,...,}`

  * 중괄호 안에 키:값의 쌍으로 된 항목을 콤마(,)로 구분하여 적어준다.

  * 키에 따옴표('')를 쓰지 않는다는 점에 주의한다.

  * 키에 따옴표('')를 쓰지 않아도 딕셔너리가 생성되면서 자동으로 문자열형으로 지정된다.

    ```python
    menu = {'김밥':2000, '떡볶이':2500, '어묵':2000, '튀김':3000}
    ```

- `dict`로 딕셔너리 만들기

  - 방법 1 : `딕셔너리명 = dict(키1=값1, 키2=값2,...,)`

    ```python
    menu1 = dict(김밥=2000, 떡볶이=2500, 어묵=2000, 튀김=3000)
    ```

  - 방법 2 : `딕셔너리명 = dict(zip(key리스트, value리스트))`

    ```python
    key_list = ['김밥','떡볶이','어묵','튀김']
    value_list = [2000,2500,2000,3000]
    menu2 = dict(zip(key_list,value_list))
    menu2
    ```

  - 방법 3 : `딕셔너리명 = dict([(키1,값1),(키1,값2),...,])`

    ```python
    [('김밥',2000),('떡볶이',2500),('어묵',2000),('튀김',3000)]
    menu3 = dict()
    menu3
    ```

  - 방법 4 : `딕셔너리명 = dict({키1:값1,키2:값2,...,})`

    ```python
    menu4 = dict({'김밥':2000, '떡볶이':2500, '어묵':2000, '튀김':3000})
    print(menu4)
    ```

<br>

1. for, if 사용

   ```python
   book_dict = {}
   for title in book_title:
       if title in book_dict:
           book_dict[title] += 1 # 기존 키 값이 있으면 1을 더함
       else:
           book_dict[title] = 1 # 기존 키 값이 없으면 1 값을 할당
   print(book_dict)
   ```

2. `count()`사용

   ```python
   book_dict = {}
   for title in book_title:
       # 리스트의 특정 요소가 몇 개 있는지 count해서 그 값을 딕셔너리의 value로 설정
       book_dict[title] = book_title.count(title)
   print(book_dict)
   ```

3. `get()`사용

   ```python
   book_dict = {}
   for title in book_title:
       # 만약 key가 없으면 None이 아닌 0을 value로 받는다.
       book_dict[title] = book_dict.get(title, 0) + 1
   print(book_dict)
   ```

<br>

<br>

#### 5-3 딕셔너리 항목 추가, 삭제하기

- 딕셔너리 항목 추가하기

  - `딕셔너리명[키]=값`
  - 키가 존재하지 않으면 추가, 존재하면 수정된다.

  ```python
  scores = {'kor':100, 'eng':90, 'math':80}
  
  # math점수 85점으로 수정하기
  scores['math'] = 85
  # music점수 95점 추가하기
  scores['music'] = 95
  scores
  # >> {'eng': 90, 'kor': 100, 'math': 85, 'music': 95}
  ```

- setdefault로 항목 추가하기

  - `딕셔너리명.setdefault(키,값)`
  - 이미 들어있는 키의 값은 수정할 수 없다.

  ```python
  scores = {'kor':100, 'eng':90, 'math':80}
  
  # music점수 넣기(key만 넣고 value를 생략하면? value에 빈값) (None)
  scores.setdefault('music')
  scores
  # {'eng': 90, 'kor': 100, 'math': 80, 'music': None}
  ```

  ```python
  # music점수 90점 추가
  scores.setdefault('music',90)
  scores
  # {'eng': 90, 'kor': 100, 'math': 80, 'music': 90}
  ```

  ```python
  # music점수 85점으로 수정
  scores.setdefault('music',85) # 변경되지 않음
  scores
  # {'eng': 90, 'kor': 100, 'math': 80, 'music': 95}
  ```

- update로 여러 항목 추가/ 수정하기

  - 키가 존재하면 수정, 존재하지 않으면 추가된다.

  - 방법 1

    - `딕셔너리명.update(키1=값1, 키1=값2,...)`
    - 키에 따옴표를 하지 않지만, 딕셔너리에 들어갈 때는 따옴표가 붙어서 들어간다.

    ```python
    scores = {'kor':100, 'eng':90, 'math':80}
    
    # math:90, music:90
    scores.update(math=90, music=90)
    scores
    # >> {'eng': 90, 'kor': 100, 'math': 90, 'music': 90}
    ```

  - 방법 2

    - `딕셔너리명.update(zip(key리스트, value리스트)`

    ```python
    # math:90, music:90
    scores.update(zip(['math','music'],[90,90]))
    # 꼭 기억해줘~~~~~
    scores
    # >> {'eng': 90, 'kor': 100, 'math': 90, 'music': 90}
    ```

  - 방법 3

    - `딕셔너리명.update([(키1,값1),(키2,값2),...])`

    ```python
    # math:90, music:90
    scores.update([('math',90),('music',90)])
    scores
    ```

  - 방법 4

    - `딕셔너리명.update({키1:값1,키2:값2,...,})`

    ```python
    # math:90, music:90
    scores.update({'math':90, 'music':90})    
    scores
    # >> {'eng': 90, 'kor': 100, 'math': 90, 'music': 90}
    ```

<br>

- 딕셔너리 요소 삭제하기

  - `del 딕셔너리명[키]`

    - 해당 키의 항목 삭제

    ```python
    del a[1]
    a
    # # key 1이 삭제됨 (1위치 X)
    # >> {2: 'b', 'name': 'pey'}
    ```

    

  - `딕셔너리명.pop(키, 기본값)`

    - 해당 키의 항목(값) 반환하고 삭제

    ```python
    scores = {'kor':100, 'eng':90, 'math':80}
    
    # 키가 'kor'인 항목의 값 받아온 후 삭제
    kor = scores.pop('kor')
    print(kor)
    print(scores)
    # >> 100
    # >> {'eng': 90, 'math': 80}
    ```

    - 해당 키의 항목 반환하고 삭제(키가 존재하지 않을 때 기본값 반환)

    ```python
    scores = {'kor':100, 'eng':90, 'math':80}
    
    # 키가 'music'인 항목의 값 받아온 후 삭제(삭제할 키가 존재하지 않는 경우)
    music = scores.pop('music','x')
    print(music)
    print(scores)
    # >> x
    # >> {'kor': 100, 'eng': 90, 'math': 80}
    ```

  - `딕셔너리명.clear()`

    - 딕셔너리의 모든 항목 삭제

    ```python
    scores = {'kor':100, 'eng':90, 'math':80}
    scores.clear()
    print(scores)
    # >> {}
    ```

<br>

<br>

#### 5-4 딕셔너리 관련 함수

- Key 리스트 만들기 : `keys()`

  - 딕셔너리의 키만 리스트로 가져오기

  - dict_keys객체로 받아온다. 리스트처럼 사용할 수 있지만 리스트는 아니다.

  - 반환 값의 리스트 : `list(a.keys())`

  - 키 정렬하기

    ```python
    scores = {'kor':100, 'eng':90, 'math':80}
    
    sorted(scores.keys())  # 알파벳 오름차순 정렬
    # >> ['eng', 'kor', 'math']
    
    # .sort()는 지원하지 않음
    # scores.keys.sort()
    ```

<br>

- Value 리스트 만들기 : `values()`

  - 딕셔너리의 value만 리스트로 가져오기
  - dict_values객체로 받아온다.

- Key, Value 쌍 얻기 : `items()`

  - 매우 자주 사용!!!
  - 딕셔너리의 (key,value) 쌍을 리스트로 가져오기
  - dict_items객체로 받아온다.

- Key로 Value 얻기 : `get()`

  - `딕셔너리명.get(key, msg)`

    - 존재하지 않는 key로 추출 시도해도 오류가 발생하지 않는다.
    - 존재하지 않는 key로 추출 시도할 경우 출력할 메시지를 설정할 수 있다.

    ```python
    # get : key로 value 얻기
    # 딕셔너리 안에 찾으려는 key가 없을 때 미리 정해 둔 디폴트 값 반환
    print(a.get('foo','bar'))
    ```

- 해당 Key가 딕셔너리 안에 있는지 조사하기 : `in()`

  ```python
  # in : 해당 key가 딕셔너리 안에 있는지 조사하기
  print('name' in a)
  ```

<br>

<br>

<br>

### 6. 집합

#### 6-1. 집합이란?

- 집합(set)은 파이썬 2.3부터 지원하기 시작한 자료형으로, 집합에 관련된 것을 쉽게 처리하기 위해 만든 자료형이다.
- `s = set()`
- 특징
  - 중복을 허용하지 않는다.
  - 순서가 없다(Unordered).
- set 자료형에 저장된 값을 인덱싱으로 접근하려면?
  - 리스트와 튜플로 변환한 후 해야 한다.
- 중복을 허용하지 않기 때문에 자료형의 중복을 제거하기 위한 필터 역할로 종종 사용

<br>

<br>

#### 6-2. 교집합, 합집합, 차집합 구하기

- 교집합 : `&` / `intersection()`

  ```python
  # 교집합
  print(s1 & s2)  # = s1.intersection(s2)
  ```

- 합집합 : `|` / `union()`

  ```python
  # 합집합
  print(s1 | s2)  # = s1.union(s2)
  ```

- 차집합 : `-` / `difference()`

  ```python
  # 차집합
  print(s1 - s2)  # = s1.difference(s2)
  print(s2 - s2)  # = s2.difference(s1)
  ```

<br>

<br>

#### 6-3. 집합 자료형 관련 함수들

- 값 1개 추가하기 : `add()`

  ```python
  # add : 값 1개 추가
  s1 = set([1,2,3])
  s1.add(4)
  ```

- 값 여러 개 추가하기 : `update()`

  ```python
  # upadate : 값 여러 개 추가
  s1.update([4,5,6])
  ```

- 특정 값 제거하기 : `remove()`

  ```python
  # remove : 특정 값 제거
  s1.remove(2)
  ```

<br>

<br>

### 7. 불

불(bool) 자료형이란 참(True)과 거짓(False)을 나타내는 자료형이다. 불 자료형은 다음 2가지 값만을 가질 수 있다.

- True - 참
- False - 거짓

<br>

- `1 == 1` 은 "1과 1이 같은가?"를 묻는 조건문

| 값        | 참 or 거짓 |
| :-------- | :--------- |
| "python"  | 참         |
| ""        | 거짓       |
| [1, 2, 3] | 참         |
| []        | 거짓       |
| ()        | 거짓       |
| {}        | 거짓       |
| 1         | 참         |
| 0         | 거짓       |
| None      | 거짓       |



### 8. 자료형의 값을 저장

- 변수가 가리키는 메모리의 주소

  - `id()`
  - id 함수는 변수가 가리키고 있는 객체의 주소 값을 돌려주는 파이썬 내장 함수

  ```python
  a = [1,2,3]
  id(a)
  # >> 4303029896
  ```

  

- 리스트 복사

  - `b = a`

    - `id(a)` = `id(b)`
    - a가 가리키는 대상과 b가 가리키는 대상이 동일함
    - a를 바꾸면 b도 똑같이 바뀜

  - `[:]`이용

    ```python
    a = [1,2,3]
    b = a[:]
    a[1] = 4
    a
    # >>> [1,4,3]
    b
    # >>> [1,2,3]
    ```

    - a 리스트 값을 바꿔도 b 리스트에 영향 X

  - copy() 함수 이용

    ```python
    a = [1,2,3]
    b = copy(a)
    ```

    - `b is a` >> `False`
    - b와 a가 가리키는 객체는 서로 다름

  