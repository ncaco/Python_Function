import os
from pydub import AudioSegment

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
    
    # 밀리초 단위로 변환
    interval_ms = interval_seconds * 1000
    
    # 분할 작업
    total_length = len(audio)
    num_segments = (total_length // interval_ms) + (1 if total_length % interval_ms > 0 else 0)
    
    output_files = []
    
    for i in range(num_segments):
        start_time = i * interval_ms
        end_time = min((i + 1) * interval_ms, total_length)
        
        segment = audio[start_time:end_time]
        
        # 파일명 생성 (001, 002, ... 형식)
        segment_filename = f"{file_name}_{i+1:03d}{file_ext}"
        segment_path = os.path.join(output_dir, segment_filename)
        
        # 파일 저장
        segment.export(segment_path, format=file_ext.replace('.', ''))
        output_files.append(segment_path)
    
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

# 사용 예시
if __name__ == "__main__":
    # 예시 코드
    input_file = "example.mp3"  # 분할할 오디오 파일 경로
    output_directory = "output"  # 출력 디렉토리
    split_interval = 60  # 60초 간격으로 분할
    
    try:
        # 오디오 파일 총 재생 시간 확인
        duration = get_audio_duration(input_file)
        print(f"오디오 파일 재생 시간: {duration:.2f}초")
        
        # 오디오 파일 분할
        output_files = split_audio_file(input_file, output_directory, split_interval)
        print(f"총 {len(output_files)}개의 파일로 분할되었습니다:")
        for file in output_files:
            print(f" - {file}")
    except Exception as e:
        print(f"오류 발생: {str(e)}") 