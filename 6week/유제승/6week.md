## 2주차 - 차근차근 PyTorch & TorchVision - MLOps를 위한 모델 실험 추적 & 모델 웹앱 배포까지

### Part 8-2 - 7-1, 7-2

### CNN with ImageFolder & DataLoader 1 - ImageFolder 활용법

---

<br>

> ## Load "Image"

```python
from torchvision.datasets import ImageFolder
```

<br>

---

<br>

> ### 압축파일 전체 압축풀기

```python
파일명.extractall(파일위치)
```

<br>

---

<br>

> ### Tensor변수 상태의 이미지를 출력하는 방식으로 변경해주기 위해

```python
plt.imshow(img.permute(1, 2, 0)
```

: torch에서는 (차원, width, height) 순서로 되어있지만, 이미지를 출력할때는 (width, height, 차원) 순으로 변경해줘야한다.

<br>

---

<br>

> ### 사이즈의 변환이 이루어지는 부분의 코드

```python
self.conv_block_within = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 이때 사이즈의 변경이 이루어 진다
        )
        # [ 32, 10, 32, 32 ] -> [ 32, 10, 16, 16 ]
```

<br>

---

<br>

> ### 사이즈 변환 확인

### 모델 생성

```python
def forward(self, x):

        x = self.conv_block_entrance(x)
        print(x.shape)

        x = self.conv_block_within(x)
        print(x.shape)

        x = self.classifier_block(x)
        print(x.shape)
```

### 결과

```python
temp_batch_x = next(iter(train_dataloader))[0]

print(temp_batch_x.shape)

model(temp_batch_x)
```

<br>

---

<br>

> ### 연산의 흐름 확인

```python
from touchinfo import summary

summary(model, input_size[32, 3, 64, 64])
```

단, torchinfo가 설치되어 있지 않으면 <b>pip install torchinfo</b>

<br>

---

<br>

> ### 실습용 코드 수정 부분

```python
loss_fn = nn.CrossEntropyLoss() # Softmax + CrossEntropy (built-in Softmax)

optimizer = torch.optim.Adam(params=model.parameters(), # "parameters" to optimize (apply gradient descent)
                             lr=0.001)                  # "l"earning "r"ate

metric_accuracy = Accuracy(task='multiclass',num_classes=3).to(device) # from torchmetrics import Accuracy
```
