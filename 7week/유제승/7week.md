## 차근차근 PyTorch & TorchVision - MLOps를 위한 모델 실험 추적 & 모델 웹앱 배포까지

### Part 8-2 - 7-3, 7-4

### CNN with ImageFolder & DataLoader 3,4 - ImageFolder 활용법

---

> ### 확률적으로 이미지 돌리는 코드

```python
transforms.RandomHorizontalFlip(p=0.5)
# 여기서 p=0.5 는 확률을 뜻함
```

<br>

---

### conv_block_withn을 여러개 만들어도 모델의 복잡도가 늘어나는 것이 아니다. -> 여러 블럭을 만들고 순차적으로 쌓아줘야하는거다.

---

<br>

> ### out_channels 설정시 주의하기!!!

```python
out_channels=num_filters, # num_filters == num of feature-maps == num of output channels
```

<br>

---
