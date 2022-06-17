import os

T = str(int(input()))
TT = str(int(input()))
TTT = str(int(input()))

for tc in range(1, T+1):
    answer = ''
    if answer == '':
        answer = True
    elif answer != '':
        answer = False
    
    print('#{} {}' .format(tc,answer))

    
    # 코드 이동 : 원하는 곳을 블로킹 > alt 키보드 아래방향
    # 코드 복사 : alt shift 방향키
    # 코드 동시 작업 : alt + 마우스 커서 원하는 곳 누르면 같은 작업 가능
    # ctrl shift p : 설정창