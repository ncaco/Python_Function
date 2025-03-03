# STT (Speech-to-Text) 모델

이 프로젝트는 OpenAI의 Whisper 모델을 사용하여 음성 파일을 텍스트로 변환하는 기능을 제공합니다.

## 기능

- 다양한 오디오 파일 형식(WAV, FLAC, MP3 등) 지원
- 다양한 크기의 Whisper 모델 지원 (tiny, base, small, medium, large-v3)
- 한국어 음성 인식에 최적화
- 모델 캐싱으로 빠른 로딩 시간
- GUI 파일 선택기 또는 명령줄 인자를 통한 파일 선택

## 설치

1. 필요한 패키지 설치:

```bash
python setup.py
```

2. 또는 수동으로 설치:

```bash
pip install torch librosa numpy transformers
```

## 사용 방법

### 기본 사용

```bash
python sttModel.py
```

이 명령은 GUI 파일 선택기를 열어 오디오 파일을 선택할 수 있게 합니다.

### 명령줄 옵션

```bash
# 특정 파일 변환
python sttModel.py --file path/to/audio.wav

# 다른 모델 크기 사용
python sttModel.py --model openai/whisper-tiny

# 모델 캐시 사용하지 않기
python sttModel.py --no-cache

# 모델 저장 위치 변경
python sttModel.py --model_dir D:/my_models
```

### 모델 크기 옵션

- `openai/whisper-tiny`: 가장 작고 빠르지만 정확도가 낮음
- `openai/whisper-base`: 작고 빠르며 적당한 정확도
- `openai/whisper-small`: 중간 크기와 정확도 (기본값)
- `openai/whisper-medium`: 큰 크기와 높은 정확도
- `openai/whisper-large-v3`: 가장 크고 정확하지만 가장 느림

## 문제 해결

### 파일 선택 창이 보이지 않는 경우

1. 작업 표시줄이나 다른 창 뒤에 숨어있을 수 있으니 확인해보세요.
2. 그래도 보이지 않으면 콘솔에서 직접 파일 경로를 입력할 수 있습니다.
3. 또는 명령줄에서 `--file` 옵션을 사용하여 직접 파일 경로를 지정할 수 있습니다.

### 오디오 파일 로드 오류

1. WAV 형식의 파일을 사용해보세요.
2. 오디오 파일이 손상되지 않았는지 확인하세요.
3. 다른 샘플링 레이트의 파일을 시도해보세요.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 