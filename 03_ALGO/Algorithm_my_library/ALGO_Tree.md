# 트리(Tree)

- 루트 노드(root node) : 부모가 없는 최상위 노드     >> A
- 단말 노드(leaf node) : 자식이 없는 노드    >> E F G
- 크기 (size) : 트리에 포함된 모든 노드의 개수    >> 7
- 깊이 (depth) : 루트 노드부터의 거리     >> A : 0, B,C : 1, DEF : 2, G:3
- 높이 (height) : 깊이 중 최댓값    >>  3
- 차수 (degree) : 각 노드의 (자식 방향) 간선 개수    >> A번 자식 수 2, B번 자식 수 2...
- 트리의 크기가 **N**일 때, 전체 간선의 개수는 **N-1**개

![image-20220625190831866](ALGO_Tree-imgaes/image-20220625190831866.png)





## 이진 탐색 트리(Binary Search Tree)

- **왼쪽 자식 노드 < 부모 노드 < 오른쪽 자식 노드**

![image-20220625191250531](ALGO_Tree-imgaes/image-20220625191250531.png)

- 찾고자 하는 원소 : 37

- [Step 1] : 루트 노드부터 방문하여 탐색 진행 >> <u>30</u>
  - 1) 현재 노드와 찾는 원소 37 비교
    2) 찾는 원소가 더 크므로 오른쪽 방문
- [Step 2] : 현재 노드와 값을 비교 >> <u>48</u>
  - 1) 현재 노드와 찾는 원소 37 비교
    2) 찾는 원소가 더 작으므로 왼쪽 방문
- [Step 3] : 현재 노드와 값을 비교 >> <u>37</u>
  - 1. 현재 노드와 찾는 원소 37 비교
    2. 원소를 찾았으므로 탐색을 종료



## 트리의 순회(Tree Traversal)

- 트리 자료구조에 포함된 노드를 특정한 방법으로 한 번씩 방문하는 방법
- 대표적인 트리 순회 방법
  - 전위 순회(pre-order traverse) : 루트를 먼저 방문
  - 중위 순회(in-order traverse) : 왼쪽 자식을 방문한 뒤에 루트를 방문
  - 후위 순회(post-order traverse) : 오른쪽 자식을 방문한 뒤에 루트를 방문

![image-20220625191752158](ALGO_Tree-imgaes/image-20220625191752158.png)

```
[입력 예시]
7
ABC
BDE
CFG
D None None
E None None
F None None
G None None
```

- 전위 순회 : A-B-D-E-C-F-G

- 중위 순회 : D-B-E-A-F-C-G

- 후위 순회 : D-E-B-F-G-C-A

```python
# 트리의 순회(Tree Traversal) 구현 예제
class Node:
    def __init__(self, data, left_node, right_node):
        self.data = data
        self.left_node = left_node
        self.right_node = right_node
    
# 전위 순회(Preorder Traversal)
def pre_order(node):
    print(node.data, end=' ')  # 자기 자신의 데이터를 먼저 처리한 뒤 왼쪽, 오른쪽 노드
    if node.left_node != None:
        pre_order(tree[node.left_node])
    if node.right_node != None:
        pre_order(tree[node.right_node])
        
# 중위 순회(Inorder Traversal)
def in_order(node):
    if node.left_node != None:  # 왼쪽 노드를 방문한 뒤
        in_order(tree[node.left_node])
    print(node.data, end=' ')  # 자기 자신 처리하고, 오른쪽 노드 방문
    if node.right_node != None:
        in_order(tree[node.right_node])

# 후위 순회(Postorder Traversal)
def post_order(node):
    if node.left_node != None:
        post_order(tree[node.left_node])
    if node.right_node != None:
        post_order(tree[node.right_node])
    print(node.data, end = ' ')  # 왼쪽, 오른쪽 노드를 먼저 방문한 뒤에 자기 자신 처리

n = int(input())
tree = {}  # 딕셔너리로 받음

for i in range(n):
    data, left_node, right_node = input().split()
    if left_node == 'None':
        left_node = None
    if right_node == 'None':
        right_node = None
    tree[data] = Node(data, left_node, right_node) 
    
pre_order(tree['A'])
print()
in_order(tree['A'])
print()
post_order(tree['A'])
```

