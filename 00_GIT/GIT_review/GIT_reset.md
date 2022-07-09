공부한 거 올리고 하다보니 API Key를 올려버리는 실수를 저질렀다

자칫하다가는 계정을 정지하거나 API 키를 재발급 받아야 할 수도 있고 비밀번호를 바꾸라 할 수도 있다고 했다..

내 키가 아니라 수업 때 사용했던 강사님 키라 큰 민폐가 될 뻔한 상황이었다



그래서 파일을 수정하고 다시 push 했더니 history에는 그대로 남아있게 된다는 사실을 알았다

이 히스토리를 어떻게 지워야 하나 모르겠어서 github에 올라간 파일 그 시점 전으로 되돌리기를 했다



1. 돌아가고자 하는 지점 찾기 `git log`
2. 리셋하기 `git reset --hard  '돌아가고자 하는 시점의 commit'`

​       HEAD is now at .. 이 나오면 성공

3. `git push -f origin mater` 로 저장



- git reset
  - commit 상태를 원하는 시점으로 되돌리기
  - pointer 기준으로 노드끼리 연결을 끊어냄
  - 끊어낸 후 지정 commit 이후의 파일은 삭제됨



- git reset 옵션
  - `hard` : 주의!! 다시 되돌릴 수 없음
    - 지정된 commit 이후의 기록과 파일 모두 지움
  - `mixed` (default)
    - commit 기록은 삭제하되, 기존 파일 untracked 상태로 제공
    - `git reset` == `git reset --mixed` 
  - `soft`
    - commit 기록은 삭제되지만, 파일은 staged 상태로 남음



- `git push --force`
  - force 옵션으로 강제로 push
  - 혼자 쓰는 branch면 마음대로 해도 되지만
  - 공유 중인 branch이면 주의해서 사용



다행히 며칠 밖에 안된 상황이었고 업무 상황이 아닌 개인 기록용이여서 다행인 상황이었다. 회사 업무 중 이런 상황이었더라면 ...;;;; 

이런 실수를 다신 안저지르도록 보안에 좀더 신경써야겠다. 그리고 gitignore 사용하기 꼭!