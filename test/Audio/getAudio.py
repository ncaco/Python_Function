import numpy as np
from typing import Dict, Any
import tkinter as tk
from tkinter import filedialog
import os
import sys
import librosa

def select_audio_file() -> str:
    """
    파일 선택 다이얼로그를 열어 오디오 파일을 선택하는 함수
    
    Returns:
        str: 선택된 파일 경로
    """
    try:
        root = tk.Tk()
        root.withdraw()  # 메인 윈도우 숨기기
        
        # 윈도우를 최상위로 설정
        root.attributes('-topmost', True)
        
        # 파일 다이얼로그 열기
        file_path = filedialog.askopenfilename(
            title="오디오 파일 선택",
            filetypes=[
                ("오디오 파일", "*.wav *.flac *.mp3"),
                ("모든 파일", "*.*")
            ],
            parent=root  # 부모 윈도우 설정
        )
        
        # 선택 완료 후 root 윈도우 파괴
        root.destroy()
        
        # 파일이 선택되지 않았거나 tkinter 오류가 발생한 경우 콘솔 입력으로 대체
        if not file_path:
            return console_select_audio_file()
            
        return file_path
    except Exception as e:
        print(f"GUI 파일 선택기 오류: {str(e)}")
        print("콘솔 입력으로 전환합니다.")
        return console_select_audio_file()

def console_select_audio_file() -> str:
    """
    콘솔에서 직접 파일 경로를 입력받는 함수
    
    Returns:
        str: 입력된 파일 경로
    """
    print("\n파일 경로를 직접 입력하세요 (취소하려면 빈 값 입력):")
    file_path = input("> ").strip()
    
    if not file_path:
        return ""
    
    # 경로가 존재하는지 확인
    if not os.path.exists(file_path):
        print(f"오류: 파일이 존재하지 않습니다: {file_path}")
        return console_select_audio_file()
    
    return file_path

def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    오디오 파일의 정보를 가져오는 함수
    
    Args:
        file_path (str): 오디오 파일 경로
        
    Returns:
        Dict[str, Any]: 오디오 파일 정보를 담은 딕셔너리
            - duration: 오디오 길이 (초)
            - sample_rate: 샘플링 레이트
            - channels: 채널 수
    """
    try:
        # librosa를 사용하여 오디오 파일 로드
        data, samplerate = librosa.load(file_path, sr=None, mono=False)
        
        # 기본 정보 추출
        duration = librosa.get_duration(y=data, sr=samplerate)
        
        # 채널 수 확인
        if len(data.shape) > 1:
            channels = data.shape[0]
        else:
            channels = 1
        
        return {
            'duration': duration,
            'sample_rate': samplerate,
            'channels': channels,
        }
        
    except Exception as e:
        raise Exception(f"오디오 파일 정보 추출 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    # 파일 선택기로 오디오 파일 선택
    test_file = select_audio_file()
    
    if test_file:
        try:
            audio_info = get_audio_info(test_file)
            print("\n선택된 파일:", test_file)
            print("\n오디오 파일 정보:")
            for key, value in audio_info.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"오류: {str(e)}")
    else:
        print("파일이 선택되지 않았습니다.")
