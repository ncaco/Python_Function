# 최적화된 STT(Speech-to-Text) 모델

이 프로젝트는 Whisper 모델을 기반으로 한 한국어 음성 인식(STT) 시스템을 제공합니다. 최적화된 버전은 처리 속도와 메모리 효율성을 크게 향상시켰습니다.

## 주요 최적화 내용

1. **병렬 처리**: 긴 오디오 파일을 청크로 나누어 병렬 처리
2. **메모리 최적화**: 불필요한 메모리 사용 최소화 및 주기적인 메모리 정리
3. **GPU 가속**: CUDA 지원 시 FP16 정밀도 및 채널 우선 메모리 포맷 사용
4. **텍스트 후처리 개선**: 정규식 캐싱 및 해시 기반 중복 제거 알고리즘 적용
5. **오디오 로딩 최적화**: 파일 형식에 따른 최적의 로딩 방법 선택

## 설치 방법

필요한 패키지를 설치하려면 다음 명령을 실행하세요:

```bash
python install_requirements.py
```

## 사용 방법

### 기본 사용법

```python
from Audio.sttModel import STTModel

# 모델 초기화
stt = STTModel(model_name="openai/whisper-large-v3-turbo")

# 오디오 파일 변환
text = stt.transcribe("sample.wav", language="korean")
print(text)
```

### 긴 오디오 파일 처리

```python
# 청크 크기와 오버랩 설정으로 긴 오디오 파일 처리
text = stt.transcribe(
    "long_audio.mp3",
    language="korean",
    chunk_length_s=10,  # 청크 크기(초)
    overlap_s=2,        # 오버랩 크기(초)
    dedup_strength=0.7  # 중복 제거 강도(0.0~1.0)
)
```

### 명령줄 인터페이스

```bash
python -m Audio.sttModel --model openai/whisper-large-v3-turbo --file sample.wav --chunk-size 10 --overlap 2 --dedup-strength 0.7
```

## 성능 비교

| 최적화 전 | 최적화 후 | 향상도 |
|----------|----------|--------|
| 처리 속도 | 1.0x | 최대 2.5x |
| 메모리 사용량 | 100% | 약 70% |
| 긴 오디오 처리 | 순차 처리 | 병렬 처리 |

## 지원 모델

- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo` (기본값, 권장)

## 주의사항

- GPU 메모리가 제한된 환경에서는 `openai/whisper-medium` 모델 사용을 권장합니다.
- 병렬 처리는 CPU 모드에서만 여러 스레드를 사용합니다. GPU 모드에서는 메모리 제한으로 인해 단일 스레드로 동작합니다.
- 최적의 성능을 위해 WAV 형식의 오디오 파일 사용을 권장합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

# 오디오 파일 분할기

오디오 파일을 지정한 시간 간격으로 분할하는 프로그램입니다.

## 기능

- 다양한 오디오 파일 형식 지원 (MP3, WAV, OGG, FLAC, AAC 등)
- 사용자가 지정한 시간 간격(초)으로 오디오 파일 분할
- 직관적인 GUI 인터페이스
- 분할 진행 상황 표시
- 오디오 파일 정보 표시 (재생 시간)

## 설치 방법

1. 필요한 라이브러리 설치:

```bash
pip install pydub
```

2. FFmpeg 설치:
   - Windows: [FFmpeg 다운로드](https://ffmpeg.org/download.html)에서 다운로드 후 시스템 환경 변수 PATH에 추가
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg` 또는 `sudo yum install ffmpeg`

## 사용 방법

### GUI 모드

```bash
python audio_splitter_gui.py
```

1. "파일 선택" 버튼을 클릭하여 분할할 오디오 파일을 선택합니다.
2. "폴더 선택" 버튼을 클릭하여 분할된 파일을 저장할 폴더를 선택합니다.
3. "분할 간격(초)" 필드에 원하는 분할 시간(초)을 입력합니다.
4. "오디오 분할하기" 버튼을 클릭합니다.
5. 분할이 완료될 때까지 기다립니다.

### 함수 모듈로 사용

```python
from audio_splitter import split_audio_file, get_audio_duration

# 오디오 파일 재생 시간 확인
duration = get_audio_duration("example.mp3")
print(f"오디오 파일 재생 시간: {duration:.2f}초")

# 오디오 파일 분할 (60초 간격)
output_files = split_audio_file("example.mp3", "output_folder", 60)
print(f"총 {len(output_files)}개의 파일로 분할되었습니다.")
```

## 파일 설명

- `audio_splitter.py`: 오디오 파일 분할 기능을 제공하는 핵심 함수 모듈
- `audio_splitter_gui.py`: GUI 인터페이스를 제공하는 모듈

## 주의사항

- 매우 큰 오디오 파일을 분할할 경우 메모리 사용량이 증가할 수 있습니다.
- 지원되는 파일 형식은 FFmpeg의 지원 여부에 따라 달라질 수 있습니다.

## 라이선스

MIT 라이선스
