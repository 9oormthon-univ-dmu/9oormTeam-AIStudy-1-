# 6 Week

## 6주차 7-1,7-2. 

### 강의 키워드: CNN, TorchVision, DataLoader, ImageFolder, zipfile

### 강의 내용

- **ZIP 파일 압축 해제**:
필요한 모듈 임포트
'import zipfile
import os
import glob
import random'

ZIP 파일 압축 해제
'with zipfile.ZipFile("Food101.zip", "r") as zip_f:
    print("Unzipping the dataset.") 
    zip_f.extractall("food101/data")'
- "Food101.zip" 파일을 읽기 모드("r")로 열기
- 압축 해제 중임을 알리는 메시지 출력
- extractall() 메소드를 사용해 모든 내용을 "food101/data" 디렉토리에 압축 해제

압축 해제된 디렉토리 구조 탐색:
'for data in os.walk("food101/data"):
    print(data, '\n')'
- os.walk()를 사용해 "food101/data" 디렉토리와 그 하위 디렉토리를 순회
- 각 단계에서 얻은 정보(현재 디렉토리 경로, 하위 디렉토리 목록, 파일 목록)를 출력

- 이 코드는 데이터셋을 압축 해제하고 그 구조를 탐색하는 데 유용합니다. 특히 대규모 데이터셋을 다룰 때 데이터의 구조를 이해하는 데 도움이 됩니다.

- **CNN 모델 구현**:

' self.conv_block_entrance = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )'
- 첫 번째 합성곱 블록을 정의합니다.
- 두 개의 3x3 합성곱 층과 ReLU 활성화 함수, 그리고 최대 풀링 층으로 구성됩니다.
- 입력 크기 [32, 3, 64, 64]를 [32, 10, 32, 32]로 변환합니다.

'self.conv_block_within = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )'
- 두 번째 합성곱 블록을 정의합니다.
- 구조는 첫 번째 블록과 유사하지만, 채널 수가 일정하게 유지됩니다.
- 입력 크기 [32, 10, 32, 32]를 [32, 10, 16, 16]으로 변환합니다.

'self.classifier_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_filters * 16 * 16, out_features=num_classes)
        )'
- 분류기 블록을 정의합니다.
- 입력을 평탄화한 후 완전 연결 층을 통해 최종 클래스 점수를 출력합니다.
- 입력 크기 [32, 10, 16, 16]를 [32, 10]으로 변환합니다.

    'def forward(self, x):
        x = self.conv_block_entrance(x)
        x = self.conv_block_within(x)
        x = self.classifier_block(x)
        return x '
- 순전파 과정을 정의합니다.
- 입력 데이터를 각 블록을 통해 순차적으로 전달합니다.
- 마지막 return 문을 한 줄로 작성하면 메모리 효율성을 높일 수 있습니다.