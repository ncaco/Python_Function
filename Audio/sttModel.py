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
        STT 모델 초기화
        
        Args:
            model_name (str): 사용할 Whisper 모델 이름
            model_dir (str): 모델을 저장할 로컬 디렉토리 경로
            use_cache (bool): 허깅페이스 캐시 사용 여부
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
                
                self.processor = WhisperProcessor.from_pretrained(
                    self.model_dir,
                    local_files_only=True
                )
                
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_dir,
                    local_files_only=True
                ).to(self.device)
                
                elapsed_time = time.time() - start_time
                print(f"모델 로드 완료 (소요 시간: {elapsed_time:.2f}초)")
            else:
                print(f"모델 다운로드 중: {model_name}")
                start_time = time.time()
                
                self.processor = WhisperProcessor.from_pretrained(model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
                
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
        
    def load_audio(self, file_path, sample_rate=16000):
        """
        오디오 파일을 로드하고 전처리
        
        Args:
            file_path (str): 오디오 파일 경로
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 전처리된 오디오 데이터
        """
        try:
            # librosa를 사용하여 오디오 파일 로드 (mono=True로 단일 채널 변환)
            print(f"오디오 파일 로드 중: {file_path}")
            audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
            print(f"오디오 파일 로드 완료: 길이 {len(audio)/sample_rate:.2f}초")
            return audio
        except Exception as e:
            print(f"오디오 파일 로드 중 오류 발생: {str(e)}")
            
            # 대체 방법 시도
            try:
                print("대체 방법으로 오디오 파일 로드 시도...")
                import soundfile as sf
                audio, sr = sf.read(file_path)
                
                # 스테레오를 모노로 변환
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # 샘플링 레이트 변환
                if sr != sample_rate:
                    print(f"샘플링 레이트 변환: {sr} -> {sample_rate}")
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                
                print(f"오디오 파일 로드 완료: 길이 {len(audio)/sample_rate:.2f}초")
                return audio
            except Exception as e2:
                print(f"대체 방법으로도 오디오 파일 로드 실패: {str(e2)}")
                print("다른 형식의 오디오 파일을 시도해보세요 (WAV 형식 권장).")
                raise Exception(f"오디오 파일 로드 실패: {str(e)}, {str(e2)}")
    
    def transcribe(self, audio, language="korean", chunk_length_s=30, overlap_s=5, dedup_strength=0.5):
        """
        오디오를 텍스트로 변환
        
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
        
        # 짧은 오디오 처리
        input_features = self.processor(
            audio_data, 
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
                max_new_tokens=448,
                num_beams=5,  # 빔 서치 크기
                temperature=0.0,  # 낮은 온도로 더 결정적인 출력
                repetition_penalty=1.2  # 반복 패널티
            )
        
        # 결과 디코딩
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # 후처리
        return self._post_process_text(transcription, dedup_strength)
        
    def _transcribe_long_audio(self, audio, language="korean", chunk_length_s=30):
        """
        긴 오디오 파일을 청크로 나누어 처리 (레거시 버전)
        이 함수는 하위 호환성을 위해 유지되며, 실제로는 개선된 버전을 사용합니다.
        """
        return self._transcribe_long_audio_improved(audio, language, chunk_length_s)

    def _transcribe_long_audio_improved(self, audio, language="korean", chunk_length_s=30, overlap_s=0, dedup_strength=0.5):
        """
        긴 오디오 파일을 청크로 나누어 처리 (개선된 버전)
        
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
        for i in range(0, len(audio), chunk_size - overlap_size):
            end_idx = min(i + chunk_size, len(audio))
            audio_chunks.append(audio[i:end_idx])
            if end_idx == len(audio):
                break
        
        print(f"총 {len(audio_chunks)}개의 청크로 나누어 처리합니다. (청크 크기: {chunk_length_s}초, 오버랩: {overlap_s}초)")
        
        # 각 청크를 처리하여 텍스트로 변환
        transcriptions = []
        for i, chunk in enumerate(audio_chunks):
            print(f"청크 {i+1}/{len(audio_chunks)} 처리 중...")
            
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
                    max_new_tokens=444,  # 시작 토큰 4개를 고려하여 444로 감소
                    num_beams=5,  # 빔 서치 크기
                    temperature=0.0,  # 낮은 온도로 더 결정적인 출력
                    repetition_penalty=1.2  # 반복 패널티
                )
            
            # 결과 디코딩
            chunk_transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            print(f"청크 {i+1}/{len(audio_chunks)} 처리 완료: {chunk_transcription}")
            
            transcriptions.append(chunk_transcription)
        
        # 오버랩 처리 및 중복 제거
        if len(transcriptions) > 1:
            merged_text = transcriptions[0]
            
            # 후처리
            full_transcription = self._post_process_text(merged_text, dedup_strength)
        else:
            # 청크가 하나뿐인 경우
            full_transcription = self._post_process_text(transcriptions[0], dedup_strength)
        
        return full_transcription

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
        텍스트 후처리: 중복 단어, 문장 제거 및 텍스트 정규화
        
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
        
        # 중복 단어 제거 (3번 이상 연속 반복되는 단어)
        words = text.split()
        filtered_words = []
        i = 0
        while i < len(words):
            word = words[i]
            count = 1
            while i + count < len(words) and words[i + count] == word:
                count += 1
            
            # 중복 단어 처리 (3번 이상 반복되는 경우)
            if count >= 3:
                filtered_words.append(word)  # 한 번만 추가
                i += count  # 중복된 모든 단어 건너뛰기
            else:
                filtered_words.append(word)
                i += 1
        
        # 중복 제거된 단어로 텍스트 재구성
        text = ' '.join(filtered_words)
        
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 중복 문장 제거
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 짧은 문장은 직접 비교
            if len(sentence) < 10:
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
                continue
            
            # 긴 문장은 유사도 검사
            is_duplicate = False
            for i, existing_sentence in enumerate(unique_sentences):
                # 길이가 비슷한 문장만 비교 (효율성을 위해)
                if abs(len(existing_sentence) - len(sentence)) > len(sentence) * 0.5:
                    continue
                    
                # 문장 유사도 계산 (단어 기반)
                sentence_words = set(sentence.lower().split())
                existing_words = set(existing_sentence.lower().split())
                
                # 공통 단어 수
                common_words = sentence_words.intersection(existing_words)
                
                # 유사도 계산 (Jaccard 유사도)
                similarity = len(common_words) / len(sentence_words.union(existing_words))
                
                # 중복 제거 강도에 따라 유사도 임계값 조정
                similarity_threshold = 0.5 + (dedup_strength * 0.3)  # 0.5~0.8 범위
                
                if similarity > similarity_threshold:
                    # 더 긴 문장을 유지
                    if len(sentence) > len(existing_sentence):
                        unique_sentences[i] = sentence
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        # 처리된 문장 재결합
        processed_text = ' '.join(unique_sentences)
        
        # 추가 정리
        processed_text = re.sub(r'\s+', ' ', processed_text)  # 여러 공백을 하나로
        processed_text = re.sub(r'([.!?])\s*\1+', r'\1', processed_text)  # 중복 문장 부호 제거
        processed_text = re.sub(r'^(그리고|그래서|하지만)\s+', '', processed_text)  # 문장 시작의 접속사 제거
        
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
    parser.add_argument("--chunk-size", type=int, default=30,
                        help="긴 오디오 파일을 처리할 때 사용할 청크 크기(초), 기본값: 30초")
    parser.add_argument("--overlap", type=int, default=5,
                        help="청크 간 오버랩 크기(초), 기본값: 5초")
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



