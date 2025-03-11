import torch
import librosa
import numpy as np
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, logging
from getAudio import get_audio_info, select_audio_file
import argparse
import time
import warnings
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# 불필요한 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The attention mask is not set")
warnings.filterwarnings("ignore", message="Increase max_length")

# aifc 모듈 가져오기 시도
try:
    import aifc
except ImportError:
    print("aifc 모듈을 가져올 수 없습니다. standard-aifc 패키지를 설치해보세요.")
    print("pip install standard-aifc")

# sunau 모듈 가져오기 시도 (실패해도 계속 진행)
try:
    import sunau
except ImportError:
    print("sunau 모듈을 가져올 수 없습니다. librosa를 사용하여 오디오 파일을 처리합니다.")

# 로깅 레벨 설정 (경고 메시지 숨기기)
logging.set_verbosity_error()

class STTModel:
    def __init__(self, model_name="openai/whisper-large-v3-turbo", model_dir="./models", use_cache=True):
        """
        STT 모델 초기화 (최적화 버전)
        
        Args:
            model_name (str): 사용할 Whisper 모델 이름
            model_dir (str): 모델을 저장할 로컬 디렉토리 경로
            use_cache (bool): 허깅페이스 캐시 사용 여부
        """
        # 디바이스 설정 및 최적화
        if torch.cuda.is_available():
            self.device = "cuda"
            # CUDA 최적화 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = "cpu"
            # CPU 최적화 설정
            torch.set_num_threads(min(8, os.cpu_count() or 4))
        
        print(f"사용 중인 디바이스: {self.device}")
        
        self.model_name = model_name
        self.model_dir = os.path.join(model_dir, model_name.split('/')[-1])
        
        # 모델 디렉토리가 없으면 생성
        os.makedirs(self.model_dir, exist_ok=True)
        
        try:
            # 로컬에 모델이 있는지 확인하고 로드
            if os.path.exists(os.path.join(self.model_dir, "config.json")) and use_cache:
                print(f"로컬에서 모델 로드 중: {self.model_dir}")
                start_time = time.time()
                
                # 프로세서 로드
                self.processor = WhisperProcessor.from_pretrained(
                    self.model_dir,
                    local_files_only=True
                )
                
                # 모델 로드 (메모리 최적화)
                model_kwargs = {}
                if self.device == "cuda":
                    # 가능한 경우 FP16 정밀도 사용
                    if torch.cuda.get_device_capability()[0] >= 7:  # Volta 이상의 아키텍처
                        model_kwargs["torch_dtype"] = torch.float16
                
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    **model_kwargs
                ).to(self.device)
                
                # 모델 최적화
                if self.device == "cuda":
                    self.model = self.model.to(memory_format=torch.channels_last)
                
                elapsed_time = time.time() - start_time
                print(f"모델 로드 완료 (소요 시간: {elapsed_time:.2f}초)")
            else:
                print(f"모델 다운로드 중: {model_name}")
                start_time = time.time()
                
                # 프로세서 로드
                self.processor = WhisperProcessor.from_pretrained(model_name)
                
                # 모델 로드 (메모리 최적화)
                model_kwargs = {}
                if self.device == "cuda":
                    # 가능한 경우 FP16 정밀도 사용
                    if torch.cuda.get_device_capability()[0] >= 7:  # Volta 이상의 아키텍처
                        model_kwargs["torch_dtype"] = torch.float16
                
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    **model_kwargs
                ).to(self.device)
                
                # 모델 최적화
                if self.device == "cuda":
                    self.model = self.model.to(memory_format=torch.channels_last)
                
                elapsed_time = time.time() - start_time
                print(f"모델 다운로드 완료 (소요 시간: {elapsed_time:.2f}초)")
                
                # 모델 저장
                print(f"모델 저장 중: {self.model_dir}")
                self.processor.save_pretrained(self.model_dir)
                self.model.save_pretrained(self.model_dir)
                print("모델 저장 완료")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            print("인터넷 연결을 확인하거나 다른 모델을 시도해보세요.")
            raise
        
        # 추론 최적화 설정
        self.model.eval()  # 평가 모드로 설정
        
        # 캐시 초기화
        self._cached_decoder_prompt_ids = {}
        
    def load_audio(self, file_path, sample_rate=16000):
        """
        오디오 파일을 로드하고 전처리 (최적화 버전)
        
        Args:
            file_path (str): 오디오 파일 경로
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 전처리된 오디오 데이터
        """
        try:
            # 파일 확장자 확인
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            print(f"오디오 파일 로드 중: {file_path}")
            start_time = time.time()
            
            # 파일 형식에 따라 최적의 로딩 방법 선택
            if ext in ['.wav', '.wave']:
                # WAV 파일은 scipy로 빠르게 로드
                try:
                    from scipy.io import wavfile
                    sr, audio = wavfile.read(file_path)
                    
                    # 스테레오를 모노로 변환
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1).astype(np.float32)
                    else:
                        # 정수 형식을 float32로 변환
                        if audio.dtype != np.float32:
                            max_value = np.iinfo(audio.dtype).max
                            audio = audio.astype(np.float32) / max_value
                    
                    # 샘플링 레이트 변환
                    if sr != sample_rate:
                        # librosa 대신 resampy 사용 (더 빠름)
                        try:
                            import resampy
                            audio = resampy.resample(audio, sr, sample_rate)
                        except ImportError:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                except Exception:
                    # scipy로 실패하면 librosa로 시도
                    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
            
            elif ext in ['.mp3', '.m4a', '.aac', '.ogg', '.flac']:
                # 압축 오디오 형식은 librosa 사용
                audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
            
            else:
                # 기타 형식은 librosa로 시도
                audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
            
            # 오디오 정규화 (필요한 경우)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            elapsed_time = time.time() - start_time
            print(f"오디오 파일 로드 완료: 길이 {len(audio)/sample_rate:.2f}초 (소요 시간: {elapsed_time:.2f}초)")
            return audio
        
        except Exception as e:
            print(f"오디오 파일 로드 중 오류 발생: {str(e)}")
            
            # 대체 방법 시도
            try:
                print("대체 방법으로 오디오 파일 로드 시도...")
                
                # soundfile 사용 시도
                try:
                    import soundfile as sf
                    audio, sr = sf.read(file_path)
                    
                    # 스테레오를 모노로 변환
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    
                    # 샘플링 레이트 변환
                    if sr != sample_rate:
                        print(f"샘플링 레이트 변환: {sr} -> {sample_rate}")
                        try:
                            import resampy
                            audio = resampy.resample(audio, sr, sample_rate)
                        except ImportError:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                except:
                    # ffmpeg 직접 사용 시도
                    try:
                        import subprocess
                        import tempfile
                        
                        # 임시 WAV 파일 생성
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # ffmpeg로 변환
                        cmd = [
                            'ffmpeg', '-i', file_path, 
                            '-ar', str(sample_rate), 
                            '-ac', '1', 
                            '-f', 'wav', 
                            temp_path
                        ]
                        subprocess.run(cmd, check=True, capture_output=True)
                        
                        # 변환된 파일 로드
                        from scipy.io import wavfile
                        sr, audio = wavfile.read(temp_path)
                        
                        # 정수 형식을 float32로 변환
                        if audio.dtype != np.float32:
                            max_value = np.iinfo(audio.dtype).max
                            audio = audio.astype(np.float32) / max_value
                        
                        # 임시 파일 삭제
                        os.unlink(temp_path)
                    except:
                        # 마지막 수단으로 librosa 사용
                        audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
                
                print(f"대체 방법으로 오디오 파일 로드 완료: 길이 {len(audio)/sample_rate:.2f}초")
                return audio
            except Exception as e2:
                print(f"대체 방법으로도 오디오 파일 로드 실패: {str(e2)}")
                print("다른 형식의 오디오 파일을 시도해보세요 (WAV 형식 권장).")
                raise Exception(f"오디오 파일 로드 실패: {str(e)}, {str(e2)}")
    
    def transcribe(self, audio, language="korean", chunk_length_s=5, overlap_s=0, dedup_strength=0.5):
        """
        오디오를 텍스트로 변환 (최적화 버전)
        
        Args:
            audio: 오디오 데이터 또는 파일 경로
            language: 오디오의 언어
            chunk_length_s: 청크 길이(초)
            overlap_s: 청크 간 오버랩 길이(초)
            dedup_strength: 중복 제거 강도 (0.0~1.0)
            
        Returns:
            str: 변환된 텍스트
        """
        # 오디오 데이터 로드
        if isinstance(audio, str):
            audio_data = self.load_audio(audio)
        else:
            audio_data = audio
            
        # 오디오 길이 확인
        audio_length_s = len(audio_data) / 16000
        
        # 긴 오디오 처리 여부 결정
        if audio_length_s > chunk_length_s:
            print(f"긴 오디오 파일 감지: 청크 크기 {chunk_length_s}초, 오버랩 {overlap_s}초로 처리합니다.")
            return self._transcribe_long_audio_improved(
                audio_data, 
                language=language, 
                chunk_length_s=chunk_length_s, 
                overlap_s=overlap_s,
                dedup_strength=dedup_strength
            )
        
        # 짧은 오디오 처리 (최적화)
        start_time = time.time()
        
        # 오디오 데이터를 모델 입력 형식으로 변환
        input_features = self.processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # attention_mask 생성 (모든 값이 1인 마스크)
        attention_mask = torch.ones(input_features.shape[0], input_features.shape[1], dtype=torch.long, device=self.device)
        
        # 언어 강제 지정 (캐싱 적용)
        forced_decoder_ids = None
        if language:
            if language in self._cached_decoder_prompt_ids:
                forced_decoder_ids = self._cached_decoder_prompt_ids[language]
            else:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                self._cached_decoder_prompt_ids[language] = forced_decoder_ids
        
        # 추론 실행 (최적화)
        with torch.no_grad():
            # 추론 시 메모리 최적화
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 모델 생성 파라미터 최적화
            predicted_ids = self.model.generate(
                input_features, 
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=448,
                num_beams=5,
                temperature=0.0,
                repetition_penalty=1.2,
                length_penalty=1.0,  # 길이 패널티 추가
                no_repeat_ngram_size=3  # n-gram 반복 방지
            )
        
        # 결과 디코딩
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # 메모리 정리
        del input_features, attention_mask, predicted_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        print(f"변환 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        # 후처리
        return self._post_process_text(transcription, dedup_strength)
        
    def _transcribe_long_audio(self, audio, language="korean", chunk_length_s=5):
        """
        긴 오디오 파일을 청크로 나누어 처리 (레거시 버전)
        이 함수는 하위 호환성을 위해 유지되며, 실제로는 개선된 버전을 사용합니다.
        """
        return self._transcribe_long_audio_improved(audio, language, chunk_length_s)

    def _transcribe_long_audio_improved(self, audio, language="korean", chunk_length_s=5, overlap_s=0, dedup_strength=0.5):
        """
        긴 오디오 파일을 청크로 나누어 병렬 처리 (개선된 버전)
        
        Args:
            audio (np.ndarray): 오디오 데이터
            language (str): 오디오의 언어
            chunk_length_s (int): 청크 길이(초)
            overlap_s (int): 청크 간 오버랩 길이(초)
            dedup_strength (float): 중복 제거 강도 (0.0~1.0)
            
        Returns:
            str: 변환된 텍스트
        """
        # 청크 크기 계산 (샘플 단위)
        chunk_size = chunk_length_s * 16000
        overlap_size = overlap_s * 16000
        
        # 오디오를 오버랩이 있는 청크로 나누기
        audio_chunks = []
        for i, start_idx in enumerate(range(0, len(audio), chunk_size - overlap_size)):
            end_idx = min(start_idx + chunk_size, len(audio))
            audio_chunks.append((i, audio[start_idx:end_idx]))  # 청크 번호(i)를 사용
            if end_idx == len(audio):
                break
        
        print(f"총 {len(audio_chunks)}개의 청크로 나누어 처리합니다. (청크 크기: {chunk_length_s}초, 오버랩: {overlap_s}초)")
        
        # 각 청크를 처리하는 함수
        def process_chunk(chunk_data):
            chunk_idx, chunk = chunk_data
            print(f"청크 {chunk_idx+1}/{len(audio_chunks)} 처리 중...")
            
            # 오디오 데이터를 모델 입력 형식으로 변환
            input_features = self.processor(
                chunk, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # attention_mask 생성 (모든 값이 1인 마스크)
            attention_mask = torch.ones(input_features.shape[0], input_features.shape[1], dtype=torch.long, device=self.device)
            
            # 언어 강제 지정
            forced_decoder_ids = None
            if language:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            
            # 추론 실행
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features, 
                    attention_mask=attention_mask,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=444,
                    num_beams=5,
                    temperature=0.0,
                    repetition_penalty=1.2
                )
            
            # 결과 디코딩
            chunk_transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # 메모리 정리
            del input_features, attention_mask, predicted_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"청크 {chunk_idx+1}/{len(audio_chunks)} 처리 완료")
            return chunk_idx, chunk_transcription
        
        # 병렬 처리 (GPU 메모리 제한으로 인해 최대 동시 작업 수 제한)
        max_workers = 1 if self.device == "cuda" else min(4, os.cpu_count())
        transcriptions = [None] * len(audio_chunks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 청크 처리 작업 제출
            future_to_idx = {executor.submit(process_chunk, chunk_data): chunk_data[0] for chunk_data in audio_chunks}
            
            # 결과 수집
            for future in as_completed(future_to_idx):
                chunk_idx, chunk_transcription = future.result()
                transcriptions[chunk_idx] = chunk_transcription
        
        # 청크 결과 병합 및 중복 제거
        merged_text = " ".join(transcriptions)
        
        # 메모리 정리
        del transcriptions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 후처리
        return self._post_process_text(merged_text, dedup_strength)

    def _is_sentence_boundary(self, text):
        """문장 경계인지 확인"""
        # 한국어 문장 종결 패턴
        endings = [".", "!", "?", "다.", "요.", "죠.", "니다.", "세요."]
        for ending in endings:
            if text.endswith(ending):
                return True
        return False

    def _post_process_text(self, text, dedup_strength=0.5):
        """
        텍스트 후처리: 중복 단어, 문장 제거 및 텍스트 정규화 (최적화 버전)
        
        Args:
            text (str): 처리할 텍스트
            dedup_strength (float): 중복 제거 강도 (0.0~1.0)
        
        Returns:
            str: 처리된 텍스트
        """
        if not text:
            return ""
        
        # 기본 텍스트 정리
        text = text.strip()
        
        # 중복 단어 제거 (정규식 사용 - 더 효율적)
        # 3번 이상 연속 반복되는 단어 패턴 찾기
        text = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', text)
        
        # 문장 분리 (정규식 캐싱)
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)
        
        # 중복 문장 제거 (해시 기반 접근법)
        if dedup_strength > 0:
            # 유사도 임계값 계산
            similarity_threshold = 0.5 + (dedup_strength * 0.3)  # 0.5~0.8 범위
            
            # 문장 해시맵 (빠른 검색용)
            sentence_map = {}
            unique_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # 짧은 문장은 직접 비교 (10자 미만)
                if len(sentence) < 10:
                    if sentence not in sentence_map:
                        sentence_map[sentence] = True
                        unique_sentences.append(sentence)
                    continue
                
                # 문장 핑거프린트 생성 (단어 집합)
                words = set(sentence.lower().split())
                
                # 중복 검사
                is_duplicate = False
                sentences_to_check = list(sentence_map.keys())
                
                for existing_sentence in sentences_to_check:
                    # 길이가 크게 다른 문장은 비교 생략 (효율성)
                    if abs(len(existing_sentence) - len(sentence)) > len(sentence) * 0.5:
                        continue
                    
                    # 기존 문장의 단어 집합
                    existing_words = set(existing_sentence.lower().split())
                    
                    # 공통 단어 수
                    common_words = words.intersection(existing_words)
                    
                    # Jaccard 유사도 계산
                    similarity = len(common_words) / max(1, len(words.union(existing_words)))
                    
                    if similarity > similarity_threshold:
                        # 더 긴 문장 유지
                        if len(sentence) > len(existing_sentence):
                            # 기존 문장 제거 후 새 문장 추가
                            unique_sentences.remove(existing_sentence)
                            del sentence_map[existing_sentence]
                            sentence_map[sentence] = True
                            unique_sentences.append(sentence)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    sentence_map[sentence] = True
                    unique_sentences.append(sentence)
        else:
            # 중복 제거 비활성화 시 빈 문장만 제거
            unique_sentences = [s.strip() for s in sentences if s.strip()]
        
        # 처리된 문장 재결합
        processed_text = ' '.join(unique_sentences)
        
        # 추가 정리 (정규식 패턴 미리 컴파일)
        space_pattern = re.compile(r'\s+')
        punct_pattern = re.compile(r'([.!?])\s*\1+')
        conj_pattern = re.compile(r'^(그리고|그래서|하지만)\s+')
        
        processed_text = space_pattern.sub(' ', processed_text)  # 여러 공백을 하나로
        processed_text = punct_pattern.sub(r'\1', processed_text)  # 중복 문장 부호 제거
        processed_text = conj_pattern.sub('', processed_text)  # 문장 시작의 접속사 제거
        
        return processed_text

# 사용 예시
if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="STT 모델을 사용하여 음성을 텍스트로 변환")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo", 
                        choices=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", 
                                "openai/whisper-medium", "openai/whisper-large-v3", "openai/whisper-large-v3-turbo"],
                        help="사용할 Whisper 모델 크기")
    parser.add_argument("--model_dir", type=str, default="./models", 
                        help="모델을 저장할 디렉토리 경로")
    parser.add_argument("--no-cache", action="store_true",
                        help="캐시를 사용하지 않고 항상 모델을 다운로드")
    parser.add_argument("--file", type=str, default=None,
                        help="변환할 오디오 파일 경로 (지정하지 않으면 파일 선택 다이얼로그가 열립니다)")
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="긴 오디오 파일을 처리할 때 사용할 청크 크기(초), 기본값: 5초")
    parser.add_argument("--overlap", type=int, default=0,
                        help="청크 간 오버랩 크기(초), 기본값: 0초")
    parser.add_argument("--dedup-strength", type=float, default=0.5,
                        help="중복 제거 강도 (0.0~1.0), 값이 클수록 더 적극적으로 중복 제거, 기본값: 0.5")
    parser.add_argument("--verbose", action="store_true",
                        help="상세한 로그 출력")
    args = parser.parse_args()
    
    try:
        # 상세 로그 모드 설정
        if args.verbose:
            logging.set_verbosity_info()
        
        print(f"모델 초기화 중: {args.model}")
        stt_model = STTModel(model_name=args.model, model_dir=args.model_dir, use_cache=not args.no_cache)
        
        # 오디오 파일 경로 가져오기
        audio_path = args.file
        if not audio_path:
            print("오디오 파일 선택...")
            print("파일 선택 창이 열리지 않으면 작업 표시줄이나 다른 창 뒤에 숨어있을 수 있습니다.")
            audio_path = select_audio_file()
        
        if not audio_path:
            print("오디오 파일이 선택되지 않았습니다.")
            sys.exit(1)
        
        # 오디오 파일 로드
        try:
            audio_data = stt_model.load_audio(audio_path)
        except Exception as e:
            print(f"오디오 파일 로드 중 오류 발생: {e}")
            sys.exit(1)
        
        # 음성을 텍스트로 변환
        print(f"'{audio_path}' 파일을 변환 중...")
        
        # 오디오 길이 확인
        audio_length_s = len(audio_data) / 16000
        print(f"오디오 길이: {audio_length_s:.2f}초")
        
        # 긴 오디오 처리 여부 결정
        if audio_length_s > args.chunk_size:
            print(f"긴 오디오 파일 감지: 청크 크기 {args.chunk_size}초, 오버랩 {args.overlap}초로 처리합니다.")
            print(f"중복 제거 강도: {args.dedup_strength:.1f}")
            text = stt_model._transcribe_long_audio_improved(
                audio_data, 
                chunk_length_s=args.chunk_size, 
                overlap_s=args.overlap,
                dedup_strength=args.dedup_strength
            )
        else:
            print(f"짧은 오디오 파일 감지: 단일 처리합니다. 중복 제거 강도: {args.dedup_strength:.1f}")
            text = stt_model.transcribe(
                audio_data,
                dedup_strength=args.dedup_strength
            )
        
        print("\n변환 결과:")
        print("-" * 80)
        print(text)
        print("-" * 80)
        
        # 결과 저장
        output_path = os.path.splitext(audio_path)[0] + ".txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"변환 결과가 '{output_path}'에 저장되었습니다.")
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")



