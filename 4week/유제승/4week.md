## 2주차 - 차근차근 PyTorch & TorchVision - MLOps를 위한 모델 실험 추적 & 모델 웹앱 배포까지

### Part 8-2 - 6-1, 6-2

### Torchvision에 대한 내용

<br>

```python
# 클래스 이름들 확인
train_data.classes

# 클래스 이름 인덱스 매칭 확인
train_data.class_to_idx
```

<br>

### DataLoader를 하는 이유

: 미니배치(mini-batches)를 구현하여 데이터 학습 시 더 빠른 속도로 학습하기 위해.

```python
from torch.utils.data import DataLoader
```

<br><br>

### Python에서 iterator 사용하는 법

```python
iterable = iter([1, 2, 3, 4, 5])

print(next(iterable))
print(next(iterable))

# 출력 : 1
#       2
```

<br>

### Build Model

#### Flatten Layer

```python
flatten_layer = nn.Flatten()
x = batch_X[0]
flatten_output = flatten_layer(x)

print(x.shape) # num_channels, height, width
print(flatten_output.shape) # num_channels, height * width

# 결과
# torch.Size([1, 28, 28])
# torch.Size([1, 784])
```

: 하나의 차원을 줄여서 나타내 준다.

### Train Model

> ### 실습용 수정 코드

```python
loss_fn = nn.CrossEntropyLoss() # Softmax + CrossEntropy (built-in Softmax)

optimizer = torch.optim.Adam(params=model.parameters(),  # "parameters" to optimize (apply gradient descent)
                             lr=0.01) # "l"earning "r"ate

metric_accuracy = Accuracy(task='multiclass',num_classes=10).to(device) # from torchmetrics import Accuracy
```

<br>

> 모델 이름 출력하는 코드

```python
model.__class__.__name__
```
