## 2주차 - 차근차근 PyTorch & TorchVision - MLOps를 위한 모델 실험 추적 & 모델 웹앱 배포까지

### Part 8-2 - 6-3, 6-4

### Torchvision_DataLoader 3

---

### nn.Conv2d

: 2D 입력 데이터(예: 이미지)에 대해 컨볼루션(합성곱) 연산을 수행하는 레이어

> ### 파라미터

- in_channels=3: 입력 채널의 수
- kernel_size=(3, 3): 컨볼루션 필터 크기
- stride=1: 보폭 지정
- padding=0: 딩의 양 지정

```python
conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=0)
conv_result = conv_layer(image_sample)
print('After conv_layer :', conv_result.shape)
```

<br>

---

<br>

> ### nn.MaxPool2d
>
> : 맥스 풀링(Max Pooling) 연산을 수행하여 입력 데이터의 공간적 차원을 줄임

- kernel_size=2: 풀링 윈도우의 크기를 지정

```python
pool_layer = nn.MaxPool2d(kernel_size=2)
pool_result = pool_layer(conv_result)
print('After pool_layer :', pool_result.shape)
```
