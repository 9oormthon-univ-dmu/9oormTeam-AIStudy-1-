# 8 Week

## 8주차 8 

### 강의 키워드: Modularization,Targparse,Utility

### 강의 내용

- **모델 구축 모듈 (build_model.py)**:
- 신경망 모델 정의 및 초기화
- 주요 클래스: TinyVGG
'class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        # 컨볼루션 레이어, 풀링 레이어, 선형 레이어 정의'


- **Targparse**:
- 파이썬에서 명령줄 인터페이스(CLI)를 쉽게 만들 수 있게 해주는 표준 라이브러리 모듈
- 사용자가 제공한 명령줄 인자를 파싱하고 처리하는 데 사용

- 주요 기능:
- 위치 인자(positional arguments)와 선택적 인자(optional arguments) 처리
- 자동으로 도움말과 사용법 메시지 생성
- 인자 타입 검사 및 변환
- 서브커맨드 지원

- **Utility**:
- 여러 곳에서 재사용 가능한 공통 함수들을 모아놓은 모듈
- 주로 보조적인 기능을 수행하는 함수들이 포함

- utils.py에 포함된 주요 함수들:

a. set_seeds(seed=42):
재현성을 위해 랜덤 시드를 설정하는 함수
PyTorch, NumPy, Python의 random 모듈에 대해 시드 설정
'def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)'

b. save_model(model, target_dir, model_name):
훈련된 모델을 저장하는 함수
모델의 상태 딕셔너리를 파일로 저장
'def save_model(model, target_dir, model_name):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_save_path = target_dir_path / model_name
    torch.save(obj=model.state_dict(), f=model_save_path)'

c. load_model(model, target_dir, model_name):
저장된 모델을 로드하는 함수
파일에서 모델의 상태 딕셔너리를 로드하여 모델에 적용
'def load_model(model, target_dir, model_name):
    model_load_path = Path(target_dir) / model_name
    model.load_state_dict(torch.load(f=model_load_path))
    return model'

d. plot_loss_curves(results):
훈련 및 테스트 손실과 정확도 곡선을 그리는 함수
matplotlib를 사용하여 그래프 생성
'def plot_loss_curves(results):
    # 훈련 및 테스트 손실/정확도 그래프 생성 코드'

- Utility 모듈의 장점:
- 코드 중복 감소: 여러 곳에서 사용되는 함수들을 한 곳에 모아 관리
- 유지보수 용이성: 공통 기능의 수정이 한 곳에서 이루어짐
- 코드 가독성 향상: 주요 로직과 보조 기능을 분리하여 구조화