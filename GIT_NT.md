# GIT 명령어 Cheating Sheet

##  **Git - Github 연결 (User)**

### **한번만 해도 되는 작업**

`git config --global user.name <username>` :  username 등록

`git config --global user.name` :  username 확인

`git config --global user.email <email>` : email 등록

`git config --global user.email` : email 확인



## **Git - Github 연결 (Repository)**

### **⭐️Repository 연결할때는 한번만 해도 되는 작업⭐️**

`git init` : git 시작

`git remote add origin <git repository url>` : repository 연결



### **파일 수정, 생성, 삭제 할때마다 해야하는 작업!!! == 뭐든 할때마다**

`git add .` : 모든 파일 staging area에 올리기

`git add <file_name>` :  파일 staging area에 올리기

`git commit -m '<commit message>'` : 커밋 message 작성

`git push origin master` : github로 밀어내기!



## 6/16 수업 필기

* `git config --global user.name <user_name>` : username
* `git config --global user.gmail <user_email> `: email
  * config 내 로컬과 깃헙 계정을 연결시켜줄 때 1회만!!
* `git init` : 레포를 만들고 워킹 디렉토리랑 연결시켜줄때 최초 1회
  * 레포 만들때마다!
  * 여러분들이 레포를 만들때마다 계속 해줘야 한다.
* `git status` : 워킹 디렉토리에 어떤 변화가 있는지 알아보는 명령어.
  * 워킹 디렉토리 단계와 스테이지 에리아 단계의 변화
* `git add` +`.`: 전체 다 staging area로 올리기
* `git add` + `파일명.확장자` : 이것만 올려
  * ex) `git add 파일1 파일2 파일3`
* `git commit` + `-m` `"commit message"` : 
  * 되도록 명령어로 적기
    * 동사형으로 시작하기
    * 영어로 적기
  * 약속일뿐 법은 아니다.
* `git log --oneline` : `-`(하이픈) 두개입니다.
  * `commit`된 상황에서 어떤 메시지로 언제 뭐가 올라갔는지 알기위한 명령어
* `git remote add`+`origin <github 주소>` : 내 워킹 디렉토리와 레포지토리 연결
  * `git remote -v` : 리모트가 잘 들어갔는지 확인
* `git push`+`origin master`: 최종으로 깃헙에 올린다!!



### 확인사항

1. `git config --global user.name / email` 잘 적었는지 확인

2. ```
   git remote 확인
   ```

   - `git remote add origin <깃헙 주소>`

3. `git add` 했는지

4. `git commit` 잘 했는지 확인



### 실습

1. `<username>`으로 repository 만들기
   * GIT 폴더 안에도 `<username>`이름으로 폴더 만들기
2. `README.md` 를 만들고 `자기소개 페이지 작성하기`
3. 여러번 반복하기! `add` - `commit` - `push`

