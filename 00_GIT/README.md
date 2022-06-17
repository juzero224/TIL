# markdown 필기

## Heading(#)

* `#` 해시태그의 개수만큼 헤딩의 중요도가 결정됨
* `h1`~ `h6` 태그가 있음

<br>

## List (1,*,-)

### 순서가 있는 리스트

1. 하나
2. 둘
3. 셋
4. 넷!

* `숫자`+`.`으로 작성 가능

### 순서가 없는 리스트

* 이거
* 저거
* 그거
* 요거
* `*/-` + `space bar`으로 작성 가능.

<br>

## Code Block(`)

오늘 배운 내용

```code block``` `inline code block`

```python
import time

time.sleep(60)
print('안녕 여러분?')
```

* `백킷` *3 -> 코드 블럭
* `백킷` *2 -> 인코드 블럭

<br>

## Link

[네이버](https://www.naver.com)

[구글](https://google.com)

- `[텍스트]` `(링크)`
  - 링크로 이동하게 된다.



## Image

![아이유](README-imgaes/vKspl9Et_400x400.jpg)



![꼬부기(포켓몬)](README-imgaes/1200.png)



`![텍스트]` `(링크)`

- 링크로 이동하게 된다.

<br>

## Text Emphasis(*/_)

`*/_(underbar)` 활용



### Bold

- **박주영**
- __박주영__
- **신상연**
- `**` 텍스트 `**`
- `ctrl` + `b`

### Italic

- *이수진*
- *박주영*
- `*` 텍스트 `*`
- `ctrl` + `i`

### strikeout

- ~~오지혜~~
- `~~` 텍스트 `~~`

### 형광펜

* ==텍스트==
* `==`텍스트`==`



<br>

## Divider(-/_)

---

- `_\-` * 3
- HTML
  - `<br>` : 한줄 띄어주기
  - `<hr>` : 한줄 그어주기

<hr>

<br>

## Blockquotes (>)

> `>` 한개만 작성하면 인용문을 만들 수 있습니다.
>
> > depth가 생깁니다
> >
> > > depth가 또 생겼습니다.

<br>

## Table

`|` (bar) : 선

`space bar` : 칸

| Header_1 | Header_2 |
| -------- | -------- |
| Value 01 | Value 02 |



## Check Box

* `-[ ]` : 체크박스

* `- [x]` : 체크박스 체크

- [ ] : 체크박스

- [x] : 체크박스 체크



## 위 첨자, 아래 첨자

* 텍스트 <sup>텍스트</sup>

* 텍스트 <sub>텍스트</sub>

* 텍스트 `<sup>`텍스트`</sup>`

* 텍스트 `<sub>`텍스트`</sub>`



## 각주

텍스트 Typora[^1]

[^1]: Typora는 markdown 편집기(뷰어) 입니다.

* 텍스트 Typora`[^1]`
* ^1을 누르면 각주를 넣을 수 있다.



## 주석

* 텍스트 <!--열공-->
* 텍스트 `<!--열공-->`



## HTML 태그를 이용한 글자 색상 변경

* <span style='color:blue'>안녕</span>

* <span style='color:#a83236'>나는</span>
* <span style='color:rgb(21,102,42)'>주영이야</span>

* `<span style='color:blue'>`안녕`</span>`

* `<span style='color:#a83236'>`나는`</span>`
* `<span style='color:rgb(21,102,42)'>`주영이야`</span>`

* color 검색
* [color picker](https://www.w3schools.com/colors/colors_picker.asp)

