# 5 Week

## 5주차 6-3,6-4. 

### 강의 키워드: CNN, TorchVision, DataLoader, nn.Module, Model Training, CrossEntropyLoss, Accuracy, Conv2d, MaxPool2d

### 강의 내용

- **데이터셋 로딩 및 전처리**: 
 torchvision.datasets.FashionMNIST를 사용하여 FashionMNIST 데이터셋을 로드합니다. 이 데이터는 28x28 크기의 흑백 이미지로 구성되어 있으며, 10개의 클래스로 분류됩니다. transforms.ToTensor()를 통해 데이터를 텐서로 변환하여 모델에 입력합니다.

- **DataLoader 설정**: 
 DataLoader를 사용해 데이터셋을 배치 단위로 나누어 처리하며, shuffle=True로 설정하여 데이터셋을 랜덤하게 섞습니다. 이는 모델이 데이터를 고르게 학습할 수 있도록 합니다.

- **손실 함수 및 옵티마이저 설정**: 
CrossEntropyLoss는 손실 함수로 사용되고, Adam 옵티마이저가 모델의 파라미터를 업데이트하는 데 사용됩니다.

- **Conv2d**: 
2차원 이미지에서 공간적 특징을 추출하는 역할을 합니다. 입력 이미지의 여러 특징을 필터링하며, 각 필터는 학습 과정을 통해 데이터의 중요한 패턴을 인식합니다.

주요 파라미터:
- in_channels: 입력 이미지의 채널 수. 예를 들어, 흑백 이미지는 1, 컬러 이미지는 3(RGB 채널)입니다.
- out_channels: 출력할 필터의 개수, 즉 특징 맵의 수. 이 값은 모델의 학습을 통해 출력될 특징 맵의 개수를 결정합니다.
- kernel_size: 필터의 크기. kernel_size=3은 3x3 크기의 필터를 사용한다는 뜻입니다.
- stride: 필터가 이미지를 이동하는 간격. 기본값은 1로, 한 칸씩 이동합니다.
- padding: 입력 이미지의 외곽에 추가되는 패딩. padding=1로 설정하면 이미지의 크기를 유지하며 필터링할 수 있습니다.

- **MaxPool2d**: 
공간적 다운샘플링을 수행하는 레이어로, 특징 맵의 크기를 줄이고, 중요한 특징을 보존합니다. 주로 컨볼루션 레이어 뒤에 사용되어, 모델의 복잡도를 줄이고 계산 비용을 절감합니다.

주요 파라미터:
- kernel_size: 풀링 영역의 크기. kernel_size=2는 2x2 크기의 영역에서 최대값을 추출합니다.
- stride: 풀링 영역이 이동하는 간격. 기본적으로 stride=kernel_size로 설정되어, 풀링 영역이 겹치지 않고 이동합니다.
- padding: 풀링 전 입력 데이터에 패딩을 추가할 수 있습니다. 기본값은 0입니다.
