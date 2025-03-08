# createAudio.py - 오디오 파일 생성 도구

이 도구는 다양한 유형의 WAV 오디오 파일을 생성할 수 있는 Python 스크립트입니다. 지정된 길이(초)의 무음, 순음(사인파), 백색 잡음, 주파수 스윕 등을 생성할 수 있습니다.

## 기능

- **무음 생성**: 지정된 길이의 무음 WAV 파일 생성
- **순음 생성**: 지정된 주파수와 길이의 사인파 WAV 파일 생성
- **백색 잡음 생성**: 지정된 길이의 백색 잡음 WAV 파일 생성
- **주파수 스윕 생성**: 지정된 주파수 범위와 길이의 스윕 WAV 파일 생성

## 요구 사항

- Python 3.6 이상
- NumPy
- SoundFile

다음 명령으로 필요한 패키지를 설치할 수 있습니다:

```bash
pip install numpy soundfile
```

## 사용 방법

### 명령줄 인터페이스

```bash
# 무음 생성 (5초)
python createAudio.py --type silence --duration 5

# 440Hz 순음 생성 (3초)
python createAudio.py --type tone --duration 3 --frequency 440

# 백색 잡음 생성 (2초, 진폭 0.2)
python createAudio.py --type noise --duration 2 --amplitude 0.2

# 주파수 스윕 생성 (20Hz에서 20000Hz까지, 10초)
python createAudio.py --type sweep --duration 10 --start-freq 20 --end-freq 20000

# 출력 파일 경로 지정
python createAudio.py --type tone --duration 3 --output my_tone.wav
```

### 파이썬 코드에서 사용

```python
from createAudio import create_silence, create_tone, create_white_noise, create_sweep

# 무음 생성 (5초)
silence_file = create_silence(5)

# 440Hz 순음 생성 (3초)
tone_file = create_tone(3, frequency=440)

# 백색 잡음 생성 (2초, 진폭 0.2)
noise_file = create_white_noise(2, amplitude=0.2)

# 주파수 스윕 생성 (20Hz에서 20000Hz까지, 10초)
sweep_file = create_sweep(10, start_freq=20, end_freq=20000)

# 출력 파일 경로 지정
custom_file = create_tone(3, frequency=440, output_path="my_tone.wav")
```

## 매개변수 설명

### 공통 매개변수

- `duration_seconds`: 생성할 오디오의 길이(초)
- `sample_rate`: 샘플링 레이트(Hz), 기본값 44100Hz
- `output_path`: 출력 파일 경로, 지정하지 않으면 자동 생성

### 순음 생성 매개변수

- `frequency`: 주파수(Hz), 기본값 440Hz (A4 음)
- `amplitude`: 진폭(0.0~1.0), 기본값 0.5

### 백색 잡음 생성 매개변수

- `amplitude`: 진폭(0.0~1.0), 기본값 0.1

### 주파수 스윕 생성 매개변수

- `start_freq`: 시작 주파수(Hz), 기본값 20Hz
- `end_freq`: 종료 주파수(Hz), 기본값 20000Hz
- `amplitude`: 진폭(0.0~1.0), 기본값 0.5

## 예제

### 1. 테스트 톤 생성

```python
from createAudio import create_tone

# 1kHz 테스트 톤 생성 (5초)
test_tone = create_tone(5, frequency=1000)
```

### 2. 다양한 주파수의 톤 생성

```python
from createAudio import create_tone

# 다양한 주파수의 톤 생성
frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4부터 C5까지
for i, freq in enumerate(frequencies):
    create_tone(1, frequency=freq, output_path=f"note_{i+1}.wav")
```

### 3. 오디오 테스트용 스윕 생성

```python
from createAudio import create_sweep

# 오디오 장비 테스트용 로그 스윕 생성 (20Hz에서 20kHz까지, 30초)
sweep_file = create_sweep(30, start_freq=20, end_freq=20000, output_path="audio_test_sweep.wav")
```

## 주의사항

- 생성된 파일은 기본적으로 현재 작업 디렉토리에 저장됩니다.
- 출력 파일 경로를 지정하지 않으면 타임스탬프가 포함된 파일 이름이 자동으로 생성됩니다.
- 진폭 값은 0보다 크고 1.0 이하여야 합니다. 