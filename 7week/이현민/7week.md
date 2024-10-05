# 7 Week

## 7주차 7-3,7-4. 

### 강의 키워드: Data Augmentation, Transfer Learning, Fine-tuning

### 강의 내용

- **Data Augmentation**:
- 데이터 증강을 위한 transforms 정의

'train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])'

- Resize: 이미지 크기를 224x224로 조정
- RandomHorizontalFlip: 50% 확률로 이미지를 수평으로 뒤집음
- RandomRotation: 이미지를 -45도에서 45도 사이로 무작위 회전
- ToTensor: 이미지를 텐서로 변환
- Normalize: 이미지를 정규화 (ImageNet 데이터셋의 평균과 표준편차 사용)

- **Transfer Learning**:
- 사전 학습된 ResNet18 모델 사용

'model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))'

- pretrained=True로 설정하여 ImageNet에서 사전 학습된 가중치를 가져옴
- 마지막 fully connected 층을 새로운 클래스 수에 맞게 조정
- 이를 통해 적은 양의 데이터로도 높은 성능을 달성할 수 있음

- **Fine-tuning**:
- 모델의 일부 레이어만 학습하도록 설정

'for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True'

- 모든 레이어의 가중치를 고정 (requires_grad = False)
- 마지막 fully connected 층만 학습 가능하도록 설정 (requires_grad = True)
- 이를 통해 새로운 데이터셋에 맞게 모델을 미세 조정할 수 있음

