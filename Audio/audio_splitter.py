import os
from pydub import AudioSegment
import sys
import tkinter as tk
from tkinter import filedialog

# 현재 파일의 디렉토리에서 getAudio 모듈을 직접 import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from getAudio import select_audio_file, get_audio_info

def split_audio_file(file_path, output_dir, interval_seconds):
    """
    오디오 파일을 지정된 간격으로 분할하는 함수
    
    Args:
        file_path (str): 분할할 오디오 파일 경로
        output_dir (str): 분할된 파일을 저장할 디렉토리 경로
        interval_seconds (int): 분할 간격(초)
        
    Returns:
        list: 생성된 파일 경로 리스트
    """
    # 입력 검증
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if interval_seconds <= 0:
        raise ValueError("분할 간격은 1초 이상이어야 합니다.")
    
    # 파일 확장자 확인
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # 오디오 파일 로드
        print(f"오디오 파일 로드 중: {file_path}")
        
        # ffmpeg 경로 확인
        from pydub.utils import which
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            print(f"ffmpeg 경로: {ffmpeg_path}")
        else:
            print("경고: ffmpeg를 찾을 수 없습니다. 파일 처리가 실패할 수 있습니다.")
        
        # 파일 형식에 따라 로드 방식 선택
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(file_path)
        elif file_ext == '.ogg':
            audio = AudioSegment.from_ogg(file_path)
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(file_path, "flac")
        elif file_ext == '.aac':
            audio = AudioSegment.from_file(file_path, "aac")
        else:
            audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"오디오 파일 로드 중 오류 발생: {str(e)}")
        print("이 오류는 일반적으로 ffmpeg가 설치되지 않았거나 PATH에 등록되지 않은 경우 발생합니다.")
        print("ffmpeg 설치 후 시스템을 재시작하거나 새 명령 프롬프트를 열어 다시 시도해보세요.")
        raise
    
    # 밀리초 단위로 변환
    interval_ms = interval_seconds * 1000
    
    # 분할 작업
    total_length = len(audio)
    num_segments = (total_length // interval_ms) + (1 if total_length % interval_ms > 0 else 0)
    
    print(f"오디오 길이: {total_length/1000:.2f}초, {num_segments}개 세그먼트로 분할합니다.")
    
    output_files = []
    
    for i in range(num_segments):
        start_time = i * interval_ms
        end_time = min((i + 1) * interval_ms, total_length)
        
        segment = audio[start_time:end_time]
        
        # 파일명 생성 (001, 002, ... 형식)
        segment_filename = f"{file_name}_{i+1:03d}{file_ext}"
        segment_path = os.path.join(output_dir, segment_filename)
        
        print(f"세그먼트 {i+1}/{num_segments} 저장 중: {segment_path}")
        
        try:
            # 파일 저장
            format_name = file_ext.replace('.', '')
            if format_name == 'mp3':
                segment.export(segment_path, format=format_name, bitrate="192k")
            else:
                segment.export(segment_path, format=format_name)
                
            output_files.append(segment_path)
        except Exception as e:
            print(f"세그먼트 저장 중 오류 발생: {str(e)}")
            print(f"세그먼트 {i+1}을 건너뜁니다.")
            continue
    
    return output_files

def get_audio_duration(file_path):
    """
    오디오 파일의 총 재생 시간을 초 단위로 반환하는 함수
    
    Args:
        file_path (str): 오디오 파일 경로
        
    Returns:
        float: 오디오 파일의 총 재생 시간(초)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    # 파일 확장자 확인
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        print(f"오디오 파일 길이 측정 중: {file_path}")
        
        # 오디오 파일 로드
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(file_path)
        elif file_ext == '.ogg':
            audio = AudioSegment.from_ogg(file_path)
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(file_path, "flac")
        elif file_ext == '.aac':
            audio = AudioSegment.from_file(file_path, "aac")
        else:
            audio = AudioSegment.from_file(file_path)
        
        # 밀리초를 초로 변환하여 반환
        return len(audio) / 1000.0
    except Exception as e:
        print(f"오디오 파일 길이 측정 중 오류 발생: {str(e)}")
        print("이 오류는 일반적으로 ffmpeg가 설치되지 않았거나 PATH에 등록되지 않은 경우 발생합니다.")
        print("ffmpeg 설치 후 시스템을 재시작하거나 새 명령 프롬프트를 열어 다시 시도해보세요.")
        raise

def select_output_directory() -> str:
    """
    출력 디렉토리를 선택하는 함수
    
    Returns:
        str: 선택된 디렉토리 경로
    """
    try:
        root = tk.Tk()
        root.withdraw()  # 메인 윈도우 숨기기
        
        # 윈도우를 최상위로 설정
        root.attributes('-topmost', True)
        
        # 디렉토리 다이얼로그 열기
        dir_path = filedialog.askdirectory(
            title="출력 디렉토리 선택",
            parent=root  # 부모 윈도우 설정
        )
        
        # 선택 완료 후 root 윈도우 파괴
        root.destroy()
        
        # 디렉토리가 선택되지 않은 경우 기본값 사용
        if not dir_path:
            dir_path = "output"
            print(f"디렉토리가 선택되지 않았습니다. 기본 디렉토리 '{dir_path}'를 사용합니다.")
            
            # 기본 디렉토리가 없으면 생성
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        return dir_path
    except Exception as e:
        print(f"디렉토리 선택기 오류: {str(e)}")
        print("기본 디렉토리 'output'을 사용합니다.")
        
        # 기본 디렉토리가 없으면 생성
        dir_path = "output"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        return dir_path

# 사용 예시
if __name__ == "__main__":
    # ffmpeg 확인
    try:
        from pydub.utils import which
        
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path is None:
            print("\n경고: ffmpeg가 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.")
            print("pydub 라이브러리가 오디오 처리를 위해 ffmpeg를 필요로 합니다.")
            print("\nffmpeg 설치 방법:")
            print("1. https://ffmpeg.org/download.html 에서 다운로드")
            print("2. 시스템 환경 변수 PATH에 ffmpeg 실행 파일 경로 추가")
            print("   - Windows: 제어판 > 시스템 > 고급 시스템 설정 > 환경 변수 > Path 편집")
            print("   - 일반적으로 ffmpeg.exe가 있는 bin 폴더 경로를 추가해야 합니다.")
            print("3. 또는 conda를 사용하는 경우: conda install -c conda-forge ffmpeg")
            print("4. 또는 Windows의 경우 chocolatey를 사용: choco install ffmpeg")
            print("\n설치 후 새 명령 프롬프트나 터미널을 열어 다시 시도하세요.")
            
            proceed = input("\n그래도 계속 진행하시겠습니까? (y/n): ").strip().lower()
            if proceed != 'y':
                print("프로그램을 종료합니다.")
                exit()
        else:
            print(f"ffmpeg 경로: {ffmpeg_path}")
    except Exception as e:
        print(f"ffmpeg 확인 중 오류 발생: {str(e)}")
    
    # 오디오 파일 선택
    print("\n오디오 파일을 선택해주세요.")
    input_file = select_audio_file()
    
    if not input_file:
        print("파일이 선택되지 않았습니다.")
        exit()
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"오류: 파일이 존재하지 않습니다: {input_file}")
        exit()
        
    print(f"\n선택된 파일: {input_file}")
    
    # 출력 디렉토리 선택
    print("\n출력 디렉토리를 선택해주세요.")
    output_directory = select_output_directory()
    print(f"출력 디렉토리: {output_directory}")
    
    split_interval = 60  # 60초 간격으로 분할
    
    try:
        # 오디오 파일 정보 표시
        print(f"\n선택된 파일: {input_file}")
        
        # 오디오 파일 총 재생 시간 확인
        duration = get_audio_duration(input_file)
        print(f"오디오 파일 재생 시간: {duration:.2f}초")
        
        # 상세 정보 표시
        try:
            audio_info = get_audio_info(input_file)
            print("\n오디오 파일 상세 정보:")
            print(f"샘플링 레이트: {audio_info['sample_rate']} Hz")
            print(f"채널 수: {audio_info['channels']}")
        except ModuleNotFoundError as e:
            if "librosa" in str(e):
                print("\n상세 정보 추출을 위해 librosa 라이브러리가 필요합니다.")
                print("설치 방법: pip install librosa")
                print("상세 정보 표시를 건너뜁니다.")
            else:
                print(f"상세 정보 추출 실패: {str(e)}")
        except Exception as e:
            print(f"상세 정보 추출 실패: {str(e)}")
        
        # 사용자에게 분할 간격 입력 받기
        try:
            split_interval = int(input("\n분할 간격(초)을 입력하세요 [기본값: 60초]: ") or "60")
            if split_interval <= 0:
                print("분할 간격은 1초 이상이어야 합니다. 기본값 60초로 설정합니다.")
                split_interval = 60
        except ValueError:
            print("유효한 숫자가 아닙니다. 기본값 60초로 설정합니다.")
            split_interval = 60
        
        # 오디오 파일 분할
        print(f"\n{split_interval}초 간격으로 분할을 시작합니다...")
        output_files = split_audio_file(input_file, output_directory, split_interval)
        print(f"\n분할 완료! 총 {len(output_files)}개의 파일로 분할되었습니다:")
        for file in output_files:
            print(f" - {file}")
    except Exception as e:
        print(f"오류 발생: {str(e)}") 