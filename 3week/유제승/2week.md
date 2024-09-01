## 2주차 - 차근차근 PyTorch & TorchVision - MLOps를 위한 모델 실험 추적 & 모델 웹앱 배포까지

### Part 1 다 부수기

> Preparing the dataset

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000,
                  n_features=2, # number of features
                  centers=4, # number of classes(clusters)
                  cluster_std=1.5, # 동일 클러스터 내의 데이터들을 흩어뜨리는 정도
                  random_state=42)

print(X[:5], '\n\n', y[:5])
```

> result

```shell
[[-8.41339595  6.93516545]
 [-5.76648413 -6.43117072]
 [-6.04213682 -6.76610151]
 [ 3.95083749  0.69840313]
 [ 4.25049181 -0.28154475]]

 [3 2 2 1 1]
```

### 2. Build the model (Multi-class classification)

```python
class ModelBlobMLP(nn.Module):


    def __init__(self, num_features, num_classes, num_nodes=16):

        super().__init__()

        self.sequential_stack = nn.Sequential( # Make a sequential model

            nn.Linear(in_features=2, out_features=num_nodes),
            # nn.ReLU(), # apply as needed (Non-linearity)

            nn.Linear(in_features=num_nodes, out_features=num_nodes),
            # nn.ReLU(), # apply as needed (Non-linearity)

            nn.Linear(in_features=num_nodes, out_features=4), # num_classes != number of columns in target
            # We don't need to use nn.Softmax() (check the following source codes)
        )

    def forward(self, x): # forward-pass

        return self.sequential_stack(x)
```

> ## demension : 인덱스 번호 기준 차원을 뜻 함

<br>

## argmax 함수

> ### 1. torch.argmax(TENSOR, dim=1)

```python
untrained_predictions = torch.argmax(untrained_probs, dim=1)
```

<br>

> ### 2. TENSOR.argmax(dim=1)

```python
untrained_predictions = untrained_probs.argmax(dim=1)
```

<br>

---

<br>

> # 중요한 사항
>
> ### torchmetrics 0.11.0 버전부터 중요 메트릭 클래스에서 'task' 인자를 요구하도록 변경됨.<br>따라서 'Accuracy' 인스턴스화에 'task' 인자 추가

<br>

1. 이진 분류의 경우 :

   ```python
    from torchmetrics.classification import Accuracy

    torchmetrics_accuracy = Accuracy(task='binary').to(device)
   ```

<br>

2. 다중 클래스 분류의 경우 :

   ```python
   from torchmetrics.classification import Accuracy

    num_classes = 10  # 실제 분류 클래스 수로 변경
    torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
   ```

<br>

3. 다중 레이블 분류의 경우 :

   ```python
   from torchmetrics.classification import Accracy

   num_classes = 5 # 실제 레이블 수로 변경
   torchmetrics_accuracy = Accuracy(task='multilabel', num_labels=num_labels).to(device)
   ```

<br>

---

<br>

### 3. Model 저장하기

```python
touch.save(obj=model.state_dict(), f='models/usethis_classifier.pth')
```

### 4. Model 불러오기

```python
# 모델 구조를 먼저 생성
loaded_model = MLP_Classifier(num_features=8, num_classes=2, num_nodes=256)

# pickle 파일을 먼저 load 후 parameter 값들을 꺼내어 모델로 load
loaded_model.load_state_dict(torch.load(f='models/usethis_classifier.pth'))
```

<br><br>

## Regression

```python
# scikit-learn 1.2 version 부터 Boston Data 삭제
# 데이터를 California Housing Dataset로 대체

from sklearn.datasets import fetch_california_housing

# 데이터셋 로드
california = fetch_california_housing()
```

> 따라서 5-2 강의의 모든 데이터셋 로드 부분을 **fetch_california_housing()** 로 바꿔줘야 한다.
