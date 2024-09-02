# 1 week

## 1주차 1.1. Tensor + Numpy + GPU: PyTorch의 Tensor 데이터타입 & numpy(),from_numpy()

### 강의 키워드 : Tensor , Numpy , GPU

### 1-1 Tensor 생성 및 기본 연산
- **Tensor 생성**: 다양한 방법으로 Tensor를 생성하는 방법을 다룹니다. `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.arange` 등의 함수로 Tensor를 생성하는 방법을 설명해줍니다.
- **Tensor 속성**: 생성된 Tensor의 데이터 타입, 크기, 차원 등을 확인하는 방법을 소개합니다.
- **기본 연산**: Tensor 간의 덧셈, 뺄셈, 곱셈, 나눗셈과 같은 기본적인 수학 연산을 수행하는 방법을 설명합니다. 또한, Tensor의 인덱싱과 슬라이싱에 대해 다룹니다.

## 1-2 Numpy와의 호환성
- **Numpy 변환**: PyTorch Tensor와 Numpy 배열 간의 상호 변환 방법을 설명합니다. `torch.from_numpy()`와 `.numpy()` 메서드를 사용하여 Tensor를 Numpy 배열로, 또는 그 반대로 변환하는 방법을 다룹니다.
- **메모리 공유**: PyTorch Tensor와 Numpy 배열이 메모리를 공유하는 방식과, 이를 활용하여 데이터를 효율적으로 처리하는 방법을 설명합니다.

## 1-3 GPU 사용
- **Tensor의 GPU 할당**: `torch.device`를 사용하여 Tensor를 GPU에 할당하는 방법을 설명합니다. 예를 들어, `cuda` 장치로 Tensor를 옮겨 연산을 가속화하는 방법을 다룹니다.
- **GPU 연산**: GPU에서 Tensor 연산을 수행하는 방법과 CPU와의 성능 차이를 비교합니다. Tensor를 GPU로 이동시키고, GPU에서 계산한 결과를 다시 CPU로 가져오는 방법을 소개합니다.


## 1주차 2. LinearRegression: nn.Module, nn.Parameter & PyTorch model-training process, Saving+Loading model params

### 강의 키워드 : nn.Module , nn.Parameter , PyTorch 모델 학습 과정 , 모델 저장 및 불러오기

### 2-1 nn.Module과 nn.Parameter
- **nn.Module**: PyTorch의 기본 신경망 구성 요소인 `nn.Module`에 대해 설명합니다. `nn.Module`을 사용하여 커스텀 모델을 정의하고, 모델 내에서 레이어와 파라미터를 관리하는 방법을 다룹니다.
- **nn.Parameter**: `nn.Parameter` 클래스에 대해 설명하며, 모델의 학습 가능한 파라미터로 사용할 Tensor를 정의하는 방법을 설명합니다. 이 파라미터들은 학습 중에 자동으로 업데이트됩니다.

### 2-2 PyTorch 모델 학습 과정
- **모델 정의**: 커스텀 신경망 모델을 정의하고, 해당 모델의 초기 파라미터를 설정하는 방법을 설명합니다.
- **순전파(Forward Pass)**: 입력 데이터를 모델에 통과시켜 예측값을 얻는 과정에 대해 다룹니다.
- **손실 함수(Loss Function)**: 모델의 예측값과 실제 값 사이의 차이를 계산하는 손실 함수를 정의하고 사용하는 방법을 설명합니다.
- **역전파(Backward Pass)와 최적화(Optimization)**: 손실 함수의 결과를 바탕으로 모델의 파라미터를 업데이트하는 역전파 과정과 최적화 알고리즘의 사용을 다룹니다.

### 2-3 모델 파라미터 저장 및 불러오기
- **모델 저장(Saving)**: 학습된 모델의 파라미터를 파일로 저장하는 방법을 설명합니다. `torch.save()`를 사용하여 모델의 상태 딕셔너리를 저장하는 방법을 다룹니다.
- **모델 불러오기(Loading)**: 저장된 모델 파라미터를 다시 불러와서 모델을 복원하는 방법을 설명합니다. `torch.load()`와 `load_state_dict()`를 사용하여 저장된 모델을 로드하는 방법을 다룹니다.


## 1주차 3.Binary Classification: 

## 강의 키워드 : Binary Classification , nn.Module , ogits & Sigmoid , Model Training , ReLU & Sigmoid

### 1. nn.Module과 nn.Parameter
- **nn.Module**: PyTorch의 신경망 모델을 정의하는 기본 클래스인 `nn.Module`을 사용하여 커스텀 모델을 구성하는 방법을 설명합니다. 모델 내부에서 레이어와 파라미터를 관리하는 방법에 대해 다룹니다.
- **nn.Parameter**: 학습 가능한 파라미터를 정의하기 위해 사용되는 `nn.Parameter` 클래스에 대해 설명합니다. 이 파라미터들은 학습 과정에서 자동으로 최적화됩니다.

### 2. PyTorch 모델 학습 과정
- **모델 정의**: 이진 분류 문제를 해결하기 위해 커스텀 신경망 모델을 정의하는 과정을 설명합니다. 초기 파라미터의 설정 및 모델의 구조를 설명합니다.
- **순전파(Forward Pass)**: 입력 데이터를 모델에 통과시켜 예측값을 도출하는 과정을 다룹니다.
- **손실 함수(Loss Function)**: 모델의 예측값과 실제 레이블 사이의 차이를 계산하는 손실 함수를 정의하고 이를 사용하는 방법을 설명합니다.
- **역전파(Backward Pass)와 최적화(Optimization)**: 손실 함수의 결과를 바탕으로 모델의 파라미터를 업데이트하는 역전파 과정과 이를 위해 사용하는 최적화 알고리즘에 대해 다룹니다.

### 3. 모델 파라미터 저장 및 불러오기
- **모델 저장(Saving)**: 학습이 완료된 모델의 파라미터를 파일로 저장하는 방법을 설명합니다. `torch.save()`를 사용하여 모델의 상태를 저장하는 방법을 다룹니다.
- **모델 불러오기(Loading)**: 저장된 모델의 파라미터를 다시 불러와 모델을 복원하는 과정을 설명합니다. `torch.load()`와 `load_state_dict()`를 사용하여 모델을 로드하는 방법을 다룹니다.
