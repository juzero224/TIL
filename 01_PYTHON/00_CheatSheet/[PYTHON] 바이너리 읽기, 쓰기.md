[PYTHON] 바이너리 읽기, 쓰기 

넘파이를 기준으로



바이너리(Binary) 읽기

- `numpy.frombuffer`

- 버퍼에 있는 데이터를 1차원 배열로 만들어 주는 기능

  

- `numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)`

  > buffer : 데이터 (꼭 바이너리가 아니어도 됨)
  >
  > dtype : 데이터 타입 (default : float)
  >
  > count : 읽어올 데이터 수. `-1` : 전체 값 읽어 오기
  >
  > offset : 바이너리 값을 읽어올 시작 위치. 기본 값 : 0



```python
import numpy
data = b'\xd8\x0fI@ff\xe6\x01\x00\x00\x00@'
ArrBin=numpy.frombuffer(data, dtype=numpy.float32)
print(ArrBin)

```

> [3.141592e+00 8.463559e-38 2.000000e+00]
>
> 바이너리 >> 배열





바이너리(Binary) 쓰기

- `numpy.tobytes(order)`
- 넘파이 배열에서 byte 바이너리 형태로 변환시키는 함수
- order : 다차원 배열일 경우 열과 행 중 어떤 부분을 먼저 진행할지에 대한 순서. C - Row 중심, F - Column 중심



```python
import numpy
data = b'\xd8\x0fI@ff\xe6\x01\x00\x00\x00@'
ArrBin=numpy.frombuffer(data, dtype=numpy.float32)
print(ArrBin)
print(ArrBin.tobytes())
```

> [3.141592e+00 8.463559e-38 2.000000e+00]
>
> b'\xd8\x0fI@ff\xe6\x01\x00\x00\x00@'
>
> 바이너리 형태로 값 출력





- 바이너리를 하는 이유 : 저장공간의 제약 및 데이터 처리 속도 때문