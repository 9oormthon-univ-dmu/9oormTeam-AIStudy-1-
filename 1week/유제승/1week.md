## 1주차 - 차근차근 PyTorch & TorchVision - MLOps를 위한 모델 실험 추적 & 모델 웹앱 배포까지

### 1-1 Tensor + Numpy + GPU 1

#### 변수 만들기 1차원 벡터

```python
    # [] 넣어줘서 생성해야함
    scalar = touch.Tensor([3])
```

#### 2) vector == an array == a single dimension tensor

: 1차원 벡터 만들기

#### 3) matrix == a multiple dimension tensor

: 행렬

#### 4) random / zeros / ones / arange

: 함수들

### 2. Tensor operations

:2차원 벡터 만들기

### 3. Pytorch tensor <-> Numpy array

: Tensor 에서 Numpy array로 바꾸기 <br>
-> tensor.numpy() 활용

1. Numpy array -> Pytorch tensor
   <br>
   -> form_numpy()

### Random Seed 설정

    seed=42 #seed 고정

---

### 2 LinearRegression + Model save & load

#### 1. torch.arange(0, 1, 0.02).unsqueeze(dim=1) : 차원 수 증가

#### 2. test, train 나누기

```python
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)
```

#### 3. Build the model (Linear Regression)

#### torch.**nn** <br><br> : 계산 그래프(computational graphs)를 만들기 위한 모든 building blocks (**layers**) 를 포함

<br>

<hr>
<br>

#### torch.**nn.Module** <br><br> : 모든 인공신경망 구조를 위한 기본 클래스. <br><br> PyTorch 에서 인공신경망을 만들 때에는 **언제나 nn.Module 클래스를 상속받아 모델의 클래스를 만들어야 함**<br><br>+ **"forward()" 메서드를 필수로 재정의(override)**해주어야 함

<br>

<hr>
<br>

#### torch.**nn.Parameter**<br><br> : nn.Module 에서 활용되는 tensor들을 저장 <br><br> [ **requires_grad=True** ] 가 설정되어 있을 경우 자동으로 gradients 를 계산 (== "autograd")

<br>

<hr>
<br>

#### method "**forward()**" : nn.Module 클래스를 상속받은 모든 클래스(우리가 만들 모델의 클래스)들은 forward() 메서드를 재정의(override)해주어야 함 <br><br> (forward computation에 대한 정의)

<br>

<hr>
<br>

#### torch.**optim** : 다양한 Optimizer 들을 포함

<br>

<hr>
<br>

#### 4. 순방향 전파

```python
    def foward(self, x):
        retur self.weights * x + self.bias # y = ax + b
```

#### 5. 모델의 파라미터 확인

```python
    # 단 generater(ex. list)를 꼭 적어줘야 함
    list(model.parameters())
```

#### 6. 모델을 예측 모드로 변경

```python
    with torch.inference_mode(): # Set "inference mode"
```

<br>
<hr>
<br>

#### torch.**save** <br><br> : 다양한 객체를 pickle 파일로 저장 (models, tensors, python objects like dicts)

<br>

<hr>
<br>

#### torch.**load** <br><br> : 다양한 객체를 pickle 파일로부터 로딩 (models, tensors, python objects like dicts) & 로딩 시 적재될 device도 결정 가능 (CPU, GPU, etc)

<br>

<hr>
<br>

#### torch.**nn.Module.load_state_dict** <br><br> : 저장된 **model.state_dict()** 파일로부터 모델 파라미터 값들을 로딩

<br>

<hr>
<br>

### 3. Binary Classification + nn.Sequential

#### 원형 데이터 만들기

```python
    make_circles
```

<hr>
<br>

#### torch.nn.**BCELoss**() <- should be used with sigmoid (nn.Sigmoid + nn.BCELoss)<br><br> : Binary cross-entropy

<br>

<hr>
<br>

#### torch.nn.**BCEWithLogitsLoss**() <- more numerically stable (than nn.Sigmoid + nn.BCELoss)<br><br> : Binary cross-entropy **with built-in Sigmoid**

<br>
