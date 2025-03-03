import torch
import librosa
import numpy as np
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, logging
from getAudio import get_audio_info, select_audio_file
import argparse
import time

# 로깅 레벨 설정
logging.set_verbosity_info()

class STTModel:
    def __init__(self, model_name="openai/whisper-small", model_dir="./models", use_cache=True):
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
            audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
            return audio
        except Exception as e:
            print(f"오디오 파일 로드 중 오류 발생: {str(e)}")
            print("다른 형식의 오디오 파일을 시도해보세요 (WAV 형식 권장).")
            raise
    
    def transcribe(self, file_path, language="korean"):
        """
        오디오 파일을 텍스트로 변환
        
        Args:
            file_path (str): 오디오 파일 경로
            language (str): 오디오의 언어
            
        Returns:
            str: 변환된 텍스트
        """
        # 오디오 로드
        audio = self.load_audio(file_path)
        
        # 오디오 데이터를 모델 입력 형식으로 변환
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # 언어 강제 지정 (한국어)
        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
        
        # 추론 실행
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features, 
                forced_decoder_ids=forced_decoder_ids
            )
        
        # 결과 디코딩
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription

# 사용 예시
if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="STT 모델을 사용하여 음성을 텍스트로 변환")
    parser.add_argument("--model", type=str, default="openai/whisper-small", 
                        choices=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", 
                                "openai/whisper-medium", "openai/whisper-large-v3"],
                        help="사용할 Whisper 모델 크기")
    parser.add_argument("--model_dir", type=str, default="./models", 
                        help="모델을 저장할 디렉토리 경로")
    parser.add_argument("--no-cache", action="store_true",
                        help="캐시를 사용하지 않고 항상 모델을 다운로드")
    parser.add_argument("--file", type=str, default=None,
                        help="변환할 오디오 파일 경로 (지정하지 않으면 파일 선택 다이얼로그가 열립니다)")
    args = parser.parse_args()
    
    try:
        print(f"모델 초기화 중: {args.model}")
        stt_model = STTModel(model_name=args.model, model_dir=args.model_dir, use_cache=not args.no_cache)
        
        # 파일 경로 가져오기
        file_path = args.file
        if not file_path:
            print("오디오 파일 선택...")
            print("파일 선택 창이 열리지 않으면 작업 표시줄이나 다른 창 뒤에 숨어있을 수 있습니다.")
            file_path = select_audio_file()
        
        if file_path:
            print(f"선택된 파일: {file_path}")
            
            # 파일 존재 확인
            if not os.path.exists(file_path):
                print(f"오류: 파일이 존재하지 않습니다: {file_path}")
                exit(1)
                
            print("음성을 텍스트로 변환 중...")
            start_time = time.time()
            
            text = stt_model.transcribe(file_path, language="korean")
            
            elapsed_time = time.time() - start_time
            print(f"변환 완료 (소요 시간: {elapsed_time:.2f}초)")
            print(f"\n변환된 텍스트: {text}")
        else:
            print("파일이 선택되지 않았습니다.")
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")



