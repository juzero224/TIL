# os 모듈

os 모듈은 Operating System의 약자로서 파이썬을 이용해 파일을 복사하거나 디렉터리를 생성하고 특정 디렉터리 내의 파일 목록을 구하고자 할 때 사용할 수 있다. 



자주 활용 코드

```python
import os

os.getcwd()  # 현재 경로구하기

os.listdir('경로') # 특정 경로 존재하는 파일과 디렉터리 목록 구하는 함수

files = os.listdir('c:/Anaconda3') 
len(files) # 파일 개수 확인
>>> 28
type(files)   # 파일 타입 확인
>>> <class ‘list’>

for x in os.listdir('c:/Anaconda3'): # 'exe'로 끝나는 파일만 출력
    if x.endswith('exe'):
            print(x)
```



경로와 파일명 결합 join

```python

aug_data_dir = os.path.join(data_dir, 'images') #경로와 파일명을 결합

list_path = ['C:\\', 'Users', 'user'] # 리스트로 전달가능
folder_path = os.path.join(*list_path) # 애스터리스크 * 붙이기
folder_path
->'C:\\Users\\user'
```



**파일 복사 shutil**

```python
origin='\Users\Desktop\origin\file.txt'
copy='\Users\Desktop\copy\file.txt'

shutil.copy(origin,copy) # 원래 파일 경로, 복사할 파일 경로
-> file.txt가 copy 경로에 복사되어 생성
```



파일 경로에서 확장자 분리

```python
os.path.splitext(file_path)
->('Users/Desktop/test','txt')

os.path.splitext(file_path)[1] # 확장자 알 수 있다
```



폴더 또는 경로 생성하는 법

```python
import os

# 폴더 생성
os.mkdir('./new_folder')

# 디렉토리 생성 
os.makedirs('./a/b/c', exist_ok=True)
```



mkdir

- 한 폴더만 생성 가능, makedirs처럼 폴더 내의 폴더는 생성 할 수 없음
- 기존에 new_folder라는 폴더가 있으면 os.mkdir('./new_folder/a') 를 통해 a라는 폴더 하나를 생성

makedirs

- './a/b/c' 처럼 원하는 만큼 디렉토리를 생성할 수 있습니다.
- exist_ok라는 파라미터를 True로 하면 해당 디렉토리가 기존에 존재하면 에러발생 없이 넘어가고, 없을 경우에만 생성합니다.