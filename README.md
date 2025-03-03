# Python_Function
파이썬 단순 기능 모음

## 음성 인식 (STT) 모델

Whisper 모델을 사용한 음성-텍스트 변환 기능입니다.

### 설치 방법

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

또는 직접 필요한 패키지를 설치할 수 있습니다:

```bash
pip install torch transformers soundfile numpy tkinter
```

### 사용 방법

기본 사용법:

```bash
python Audio/sttModel.py
```

이 명령은 파일 선택 대화상자를 열어 변환할 오디오 파일을 선택할 수 있게 합니다.

### 명령줄 옵션

다양한 옵션을 사용하여 STT 모델의 동작을 조정할 수 있습니다:

- `--model`: 사용할 Whisper 모델 크기 (tiny, base, small, medium, large-v3)
  ```bash
  python Audio/sttModel.py --model openai/whisper-medium
  ```

- `--file`: 변환할 오디오 파일 경로
  ```bash
  python Audio/sttModel.py --file path/to/audio.wav
  ```

- `--chunk-size`: 긴 오디오 파일을 처리할 때 사용할 청크 크기(초)
  ```bash
  python Audio/sttModel.py --chunk-size 20
  ```

- `--overlap`: 청크 간 오버랩 크기(초)
  ```bash
  python Audio/sttModel.py --overlap 10
  ```

- `--dedup-strength`: 중복 제거 강도 (0.0~1.0)
  ```bash
  python Audio/sttModel.py --dedup-strength 0.7
  ```

- `--verbose`: 상세한 로그 출력
  ```bash
  python Audio/sttModel.py --verbose
  ```

### 청크 처리 및 오버랩

긴 오디오 파일은 자동으로 청크로 나누어 처리됩니다. 청크 크기와 오버랩 크기를 조정하여 변환 품질을 향상시킬 수 있습니다:

1. **청크 크기 조정**: 기본값은 30초이며, 더 작은 값으로 설정하면 메모리 사용량이 줄어들지만 처리 시간이 길어질 수 있습니다.
   ```bash
   python Audio/sttModel.py --chunk-size 20
   ```

2. **오버랩 크기 조정**: 기본값은 5초이며, 더 큰 값으로 설정하면 청크 경계에서의 단어 분할 문제를 줄일 수 있습니다.
   ```bash
   python Audio/sttModel.py --overlap 10
   ```

3. **중복 제거 강도 조정**: 기본값은 0.5이며, 값이 클수록 더 적극적으로 중복을 제거합니다.
   ```bash
   python Audio/sttModel.py --dedup-strength 0.7
   ```

### 문제 해결

1. **텍스트가 잘리는 경우**: 오버랩 크기를 늘려보세요.
   ```bash
   python Audio/sttModel.py --overlap 10
   ```

2. **중복 텍스트가 많은 경우**: 중복 제거 강도를 높여보세요.
   ```bash
   python Audio/sttModel.py --dedup-strength 0.7
   ```

3. **중복 제거가 너무 강한 경우**: 중복 제거 강도를 낮춰보세요.
   ```bash
   python Audio/sttModel.py --dedup-strength 0.3
   ```

4. **메모리 부족 오류**: 청크 크기를 줄여보세요.
   ```bash
   python Audio/sttModel.py --chunk-size 15
   ```

5. **변환 품질이 좋지 않은 경우**: 더 큰 모델을 사용해보세요.
   ```bash
   python Audio/sttModel.py --model openai/whisper-medium
   ```

모든 옵션을 함께 사용할 수도 있습니다:

```bash
python Audio/sttModel.py --model openai/whisper-medium --chunk-size 20 --overlap 10 --dedup-strength 0.6 --file path/to/audio.wav
```
