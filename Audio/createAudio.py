import numpy as np
import soundfile as sf
import os
import argparse
from datetime import datetime
from typing import Optional, Tuple, Union

def create_silence(duration_seconds: float, 
                   sample_rate: int = 44100, 
                   output_path: Optional[str] = None) -> str:
    """
    지정된 길이의 무음 WAV 파일을 생성합니다.
    
    Args:
        duration_seconds (float): 생성할 무음의 길이(초)
        sample_rate (int, optional): 샘플링 레이트. 기본값은 44100Hz.
        output_path (str, optional): 출력 파일 경로. 지정하지 않으면 자동 생성됩니다.
        
    Returns:
        str: 생성된 WAV 파일의 경로
    """
    # 유효성 검사
    if duration_seconds <= 0:
        raise ValueError("지속 시간은 0보다 커야 합니다.")
    
    # 샘플 수 계산
    num_samples = int(duration_seconds * sample_rate)
    
    # 무음 데이터 생성 (모든 값이 0인 배열)
    silence_data = np.zeros(num_samples, dtype=np.float32)
    
    # 출력 경로가 지정되지 않은 경우 자동 생성
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"silence_{duration_seconds}sec_{timestamp}.wav"
    
    # 디렉토리가 존재하지 않으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # WAV 파일로 저장
    sf.write(output_path, silence_data, sample_rate)
    
    print(f"무음 WAV 파일이 생성되었습니다: {output_path}")
    return output_path

def create_tone(duration_seconds: float, 
                frequency: float = 440.0, 
                amplitude: float = 0.5,
                sample_rate: int = 44100, 
                output_path: Optional[str] = None) -> str:
    """
    지정된 길이와 주파수의 순음(사인파) WAV 파일을 생성합니다.
    
    Args:
        duration_seconds (float): 생성할 오디오의 길이(초)
        frequency (float, optional): 주파수(Hz). 기본값은 440Hz (A4 음).
        amplitude (float, optional): 진폭(0.0~1.0). 기본값은 0.5.
        sample_rate (int, optional): 샘플링 레이트. 기본값은 44100Hz.
        output_path (str, optional): 출력 파일 경로. 지정하지 않으면 자동 생성됩니다.
        
    Returns:
        str: 생성된 WAV 파일의 경로
    """
    # 유효성 검사
    if duration_seconds <= 0:
        raise ValueError("지속 시간은 0보다 커야 합니다.")
    if frequency <= 0:
        raise ValueError("주파수는 0보다 커야 합니다.")
    if amplitude <= 0 or amplitude > 1.0:
        raise ValueError("진폭은 0보다 크고 1.0 이하여야 합니다.")
    
    # 샘플 수 계산
    num_samples = int(duration_seconds * sample_rate)
    
    # 시간 배열 생성
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    
    # 사인파 생성
    tone_data = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # 출력 경로가 지정되지 않은 경우 자동 생성
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"tone_{frequency}hz_{duration_seconds}sec_{timestamp}.wav"
    
    # 디렉토리가 존재하지 않으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # WAV 파일로 저장
    sf.write(output_path, tone_data, sample_rate)
    
    print(f"순음 WAV 파일이 생성되었습니다: {output_path}")
    return output_path

def create_white_noise(duration_seconds: float, 
                       amplitude: float = 0.1,
                       sample_rate: int = 44100, 
                       output_path: Optional[str] = None) -> str:
    """
    지정된 길이의 백색 잡음 WAV 파일을 생성합니다.
    
    Args:
        duration_seconds (float): 생성할 오디오의 길이(초)
        amplitude (float, optional): 진폭(0.0~1.0). 기본값은 0.1.
        sample_rate (int, optional): 샘플링 레이트. 기본값은 44100Hz.
        output_path (str, optional): 출력 파일 경로. 지정하지 않으면 자동 생성됩니다.
        
    Returns:
        str: 생성된 WAV 파일의 경로
    """
    # 유효성 검사
    if duration_seconds <= 0:
        raise ValueError("지속 시간은 0보다 커야 합니다.")
    if amplitude <= 0 or amplitude > 1.0:
        raise ValueError("진폭은 0보다 크고 1.0 이하여야 합니다.")
    
    # 샘플 수 계산
    num_samples = int(duration_seconds * sample_rate)
    
    # 백색 잡음 생성
    noise_data = amplitude * np.random.uniform(-1, 1, num_samples)
    
    # 출력 경로가 지정되지 않은 경우 자동 생성
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"white_noise_{duration_seconds}sec_{timestamp}.wav"
    
    # 디렉토리가 존재하지 않으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # WAV 파일로 저장
    sf.write(output_path, noise_data, sample_rate)
    
    print(f"백색 잡음 WAV 파일이 생성되었습니다: {output_path}")
    return output_path

def create_sweep(duration_seconds: float, 
                 start_freq: float = 20.0,
                 end_freq: float = 20000.0,
                 amplitude: float = 0.5,
                 sample_rate: int = 44100, 
                 output_path: Optional[str] = None) -> str:
    """
    지정된 길이의 주파수 스윕(sweep) WAV 파일을 생성합니다.
    
    Args:
        duration_seconds (float): 생성할 오디오의 길이(초)
        start_freq (float, optional): 시작 주파수(Hz). 기본값은 20Hz.
        end_freq (float, optional): 종료 주파수(Hz). 기본값은 20000Hz.
        amplitude (float, optional): 진폭(0.0~1.0). 기본값은 0.5.
        sample_rate (int, optional): 샘플링 레이트. 기본값은 44100Hz.
        output_path (str, optional): 출력 파일 경로. 지정하지 않으면 자동 생성됩니다.
        
    Returns:
        str: 생성된 WAV 파일의 경로
    """
    # 유효성 검사
    if duration_seconds <= 0:
        raise ValueError("지속 시간은 0보다 커야 합니다.")
    if start_freq <= 0 or end_freq <= 0:
        raise ValueError("주파수는 0보다 커야 합니다.")
    if amplitude <= 0 or amplitude > 1.0:
        raise ValueError("진폭은 0보다 크고 1.0 이하여야 합니다.")
    
    # 샘플 수 계산
    num_samples = int(duration_seconds * sample_rate)
    
    # 시간 배열 생성
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    
    # 로그 스윕 생성 (지수적으로 주파수 증가)
    k = np.exp(np.log(end_freq/start_freq) / duration_seconds)
    phase = 2 * np.pi * start_freq * ((k**t - 1) / np.log(k))
    sweep_data = amplitude * np.sin(phase)
    
    # 출력 경로가 지정되지 않은 경우 자동 생성
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"sweep_{start_freq}-{end_freq}hz_{duration_seconds}sec_{timestamp}.wav"
    
    # 디렉토리가 존재하지 않으면 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # WAV 파일로 저장
    sf.write(output_path, sweep_data, sample_rate)
    
    print(f"주파수 스윕 WAV 파일이 생성되었습니다: {output_path}")
    return output_path

def main():
    """명령줄 인터페이스"""
    parser = argparse.ArgumentParser(description='오디오 파일 생성 도구')
    
    parser.add_argument('--type', type=str, default='silence',
                        choices=['silence', 'tone', 'noise', 'sweep'],
                        help='생성할 오디오 유형 (silence, tone, noise, sweep)')
    
    parser.add_argument('--duration', type=float, required=True,
                        help='오디오 길이(초)')
    
    parser.add_argument('--frequency', type=float, default=440.0,
                        help='주파수(Hz) - tone 유형에만 적용')
    
    parser.add_argument('--start-freq', type=float, default=20.0,
                        help='시작 주파수(Hz) - sweep 유형에만 적용')
    
    parser.add_argument('--end-freq', type=float, default=20000.0,
                        help='종료 주파수(Hz) - sweep 유형에만 적용')
    
    parser.add_argument('--amplitude', type=float, default=0.5,
                        help='진폭(0.0~1.0)')
    
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='샘플링 레이트(Hz)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='출력 파일 경로')
    
    args = parser.parse_args()
    
    try:
        if args.type == 'silence':
            create_silence(args.duration, args.sample_rate, args.output)
        elif args.type == 'tone':
            create_tone(args.duration, args.frequency, args.amplitude, 
                        args.sample_rate, args.output)
        elif args.type == 'noise':
            create_white_noise(args.duration, args.amplitude, 
                              args.sample_rate, args.output)
        elif args.type == 'sweep':
            create_sweep(args.duration, args.start_freq, args.end_freq, 
                        args.amplitude, args.sample_rate, args.output)
    except Exception as e:
        print(f"오류: {str(e)}")

if __name__ == "__main__":
    main()
