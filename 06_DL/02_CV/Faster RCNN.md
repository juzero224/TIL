# Faster RCNN



## 핵심

기존의 Selective Search를 제거하고 대신에 Region Proposal Network(**RPN**)을 통해서 Region of Interest(**ROI**)를 계산

- ROI : 탐지하고자 하는 구역

<br>

## 흐름

1. conv layers를 거쳐서 feature map 생성
   1. 원본 이미지를 pre trained 된 cnn 모델에 입력 → feature map 얻음
2. RPN을 거쳐서 ROI를 계산
   1. feature map → RPN → Region Proposals 산출
3. ROI Pooling 을 진행하면 고정된 크기의 feature vector 나옴
   1. 1번의 feature map + 2번의 region proposals → ROI pooling
4. 두 가지로 나뉨
   1. softmax를 통과하여 해당 ROI가 어떤 물체인지 분류 함
   2. bounding box regression을 통해 찾은 박스의 위치를 조정

![img](https://hazel-theater-7f2.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F5c2e8777-deea-4ef1-849c-18d4894e0d01%2FUntitled.png?table=block&id=646726d9-9a97-4291-bc68-9a2711f586eb&spaceId=ba727b06-1be2-49e8-ac40-4f4e0ad5a3c2&width=670&userId=&cache=v2)

<br>

<br>

## 학습

- RPN이 제대로 학습되서 ROI를 잘 계산해야 분류 레이어가 제대로 학습됨
- 4단계에 걸쳐 모델을 번갈아서 학습

1. ImageNet pretrained 모델을 불러와서 RPN을 학습 시킴

2. 기본 CNN을 제외한 Region Proposal 레이어만 가져옴

   이를 활용하여 Fast RCNN 학습.

   이 때, 처음 피처맵을 추출하는 CNN 까지 fine tune 시킴

3. 학습시킨 Fast RCNN과 RPN을 불러와서, RPN에 해당하는 레이어들만 fine tune 함

   여기서부터 RPN과 Fast RCNN이 컨볼루션 웨이트를 공유하게 됨

4. 마지막으로 공유하는 CNN과 RPN은 고정 시킨 채, Fast RCNN에 해당하는 레이어만 fine tune

<br>

<br>

## Anchor box

- 객체를 탐지하기 위한 bounding box의 일종
- 미리 정의된 3가지 scale(128, 256,512)과 3가지 aspect ratio (1:1, 1:2, 2:1) (가로세로비)를 가지는 9개의 박스

![img](https://blog.kakaocdn.net/dn/ZaxPg/btqQIaSDb3s/wfOr4FA6CxKGCgTDMkmkRk/img.jpg)

- sub sampling ratio : 원본 이미지에 grid를 나누는 비율
  - 원본 이미지 800*800, sub sampling ratio = 1/100
  - CNN 모델에 입력해 얻은 최종 feature map의 크기는 8*8 (800 * 1/100)
  - feature map의 각 cell은 원본 이미지의 100*100의 정보를 함축
- 계산 예
  - 원본 이미지 = 600*800, sub sampling ratio = 1/16
  - 생성되는 anchor 수 = 1,900 (= (600/16) * (800/16))
  - anchor box 수 = 17,100 (= 1900 * 9)

<br>

<br>

## RPN

![img](https://blog.kakaocdn.net/dn/bmPXTk/btqQKuiOcdM/zzdpQeoS1TgzfvrnfD0xKK/img.png)

1. 원본 이미지를 pre-trained된 vgg 모델에 입력하여 feature map 생성

   1. 원본 이미지의 크기 = 800*800
   2. sub-sampling rate = 1/100

   - 8x8 크기의 feature map 생성, channel 수는 512개



2. feature map에 대하여 3*3 conv 연산 적용. 크기가 유지되도록 padding 추가
   - 8x8x512 feature map에 대하여 3x3 연산 적용 → 8x8x512개의 feature map 출력



3. class score를 매기기 위해서 feature map에 대해 1x1 conv 연산 적용

   1. 이 때, channel 수가 2*9 가 되도록 설정
   2. RPN 에서는 후보 영역이 어떤 class에 해당하는지 구체적인 분류를 하지 않고, 객체 포함 여부만 분류
   3. anchor box는 각 grid cell 마다 9개가 되도록 설정
   4. channel 수는 2 (object 여부) * 9 (anchor box 개수)

   - 8x8x512 크기의 feature map을 입력 받아 8x8x2x9 크기의 feature map출력



4. bounding box regressor 를 얻기 위해서 feature map에 대해 1*1 conv 연산 적용

   1. 이 때, channel 수가 4 ( bounding box regressor) * 9 (anchor box 개수) 가 되도록 설정

   - 8x8x512 크기의 feature map을 입력 받아 8x8x4*9 크기의 feature map 출력

![img](https://blog.kakaocdn.net/dn/7Plul/btqQE9fOpFN/uK49JtrzjY7wvBZTfZhnWK/img.jpg)

- RPN 출력 결과

  - 왼쪽 : anchor box의 종류 별, 객체 포함 여부
  - 오른쪽 : anchor box의 종류 별, bounding box regressor

- 8x8 grid cell 마다 9개의 anchor box가 생성되어, 총 576(= 8*8*9) 개의 region proposals가 추출됨

- 이후 class score 에 따라 상위 n 개의 region proposals을 추출하고,

  Non maximum supression 을 적용하여 최적의 region proposals 을 Fast RCNN에 전달

<br>

<br>

## Multi-task Loss

$$
L({p_i}, {t_i}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_I^*) + \lambda \frac{1}{N_{reg}} \sum_i p_I^* L_{reg}(t_i, t_i^*)
$$

- i : mini-batch 내의 anchor의 index  
- pi : anchor ii에 객체가 포함되어 있을 예측 확률  
- p∗i : anchor가 양성일 경우 1, 음성일 경우 0을 나타내는 index parameter  
- ti : 예측 bounding box의 파라미터화된 좌표(coefficient) 
- t∗i : ground truth box의 파라미터화된 좌표 
- L_cls : Loss loss 
- L_reg : Smooth L1 loss
- N_cls : mini-batch의 크기(논문에서는 256으로 지정)
- N_reg : anchor 위치의 수
- λ : balancing parameter(default=10)



RPN 과 Fast R-CNN 학습시키기 위해 Multi-task loss 사용

RPN에서는 객체의 존재 여부만을 분류

Fast R-CNN에서는 배경을 포함한 class를 분류

<br>

<br>

## Training Faster R-CNN

### 1.feature extraction by pre-trained VGG16

- input : 800 x 800 x 3 sized image
- process : pre-trained 된 VGG16 / sub-sampling : 1/16
- output : 50 x 50 x 512 크기의 feature map



### 2. Generate Anchors by Anchor generation layer

원본 이미지에 대하여 **anchor box 생성**

grid cell 마다 9개의 anchor box 생성



1. input : 800 x 800 x 3 sized image
2. process : generate anchors
3. output : 22500(=50x50x9) (= 800 x 1/16 x 800 x 1/16 x 9) anchor box



### 3. Class scores and Bounding box regressor by RPN

VGG16으로 얻은 feature map을 입력받아 anchor에 대한 class score, bounding box regressor 반환

- input : 50x50x512 sized feature map
- process : Region proposal by RPN
- output : class scores(50x50x2x9 sized feature map) and bounding box regressor(50x50x4x9 sized feature map)



### 4. Region proposal by Proposal layer

1. Non maximum suppression을 적용하여 부적절한 객체를 제거한 후,
2. class score 상위 N개의 anchor box 추출
3. regression coefficients를 anchor box에 적용하여 anchor box가 객체의 위치를 더 잘 detect하도록 조정

- input 
  - 22500 (= 50 x 50 x 9) anchor boxes
  - class scores ( 50x50x2x9 sized feature map) and bounding box regressor (50x50x4x9 sized feature map)
- process : region proposal by proposal layer
- output : top-N ranked region proposals



### 5. Select anchors for training RPN by Anchor target layer

- RPN에 학습하는 데 사용할 수 있는 anchor 선택
- 이미지 경계를 벗어나지 않는 anchor box 선택 -> positive/negative 데이터 sampling
- ground truth box와 가장 큰 IOU 값을 가지는 경우,
- ground truth box와의 IOU 값이 0.7 이상인 경우의 box를 positive sample로 선정
- 0.3 이하인 경우 negative sample로 선정.
- 0.3~0.7 인 anchor box는 무시



### 6. Select anchors for training Fast R-CNN by Proposal Target Layer

- RPN에서 만든 region proposals 중에서 Fast RCNN 모델을 학습 시키기 위한 유용한 sample 선택
- 선택된 region proposals 는 pre-trained 모델에서 생성된 feature map에 Rol pooling을 수행

<br>

<br>

### 7. Max pooling by Rol pooling

- ex) 50*50*512 크기의 feature map + positive/negative sample
- 7*7*512 크기의 feature map

<br>

<br>

## 이후 과정

- Faster RCNN 과 같음

1. 7*7*512 크기의 feature map을 입력 받아, 4096 크기의 feature vector 얻음

2. feature vector 를 classifier와 bounding box regressor에 입력

   class의 수가 K라고 할 때, 각각 (K+1), (K+1)*4 크기의 feature vector를 출력

3. 출력된 결과를 사용하여 Multi-task loss를 통해 Fast RCNN 모델을 학습

4. loss (Loss loss + Smooth L1 loss) 출력