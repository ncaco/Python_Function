#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
createAudio.py 모듈의 기능을 테스트하는 스크립트
"""

import os
from createAudio import create_silence, create_tone, create_white_noise, create_sweep

def main():
    """
    다양한 오디오 파일 생성 함수를 테스트합니다.
    """
    print("오디오 파일 생성 테스트를 시작합니다...")
    
    # 테스트 결과를 저장할 디렉토리 생성
    test_dir = "test_audio_files"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # 1. 무음 파일 생성 테스트 (2초)
    silence_file = create_silence(
        duration_seconds=2,
        output_path=os.path.join(test_dir, "test_silence_2sec.wav")
    )
    print(f"무음 파일 생성 완료: {silence_file}")
    
    # 2. 순음 파일 생성 테스트 (440Hz, 3초)
    tone_file = create_tone(
        duration_seconds=3,
        frequency=440.0,
        amplitude=0.5,
        output_path=os.path.join(test_dir, "test_tone_440hz_3sec.wav")
    )
    print(f"순음 파일 생성 완료: {tone_file}")
    
    # 3. 백색 잡음 파일 생성 테스트 (1초)
    noise_file = create_white_noise(
        duration_seconds=1,
        amplitude=0.2,
        output_path=os.path.join(test_dir, "test_noise_1sec.wav")
    )
    print(f"백색 잡음 파일 생성 완료: {noise_file}")
    
    # 4. 주파수 스윕 파일 생성 테스트 (100Hz에서 8000Hz까지, 5초)
    sweep_file = create_sweep(
        duration_seconds=5,
        start_freq=100.0,
        end_freq=8000.0,
        amplitude=0.5,
        output_path=os.path.join(test_dir, "test_sweep_100-8000hz_5sec.wav")
    )
    print(f"주파수 스윕 파일 생성 완료: {sweep_file}")
    
    # 5. 다양한 주파수의 톤 생성 (C 메이저 스케일)
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4부터 C5까지
    scale_files = []
    
    for i, freq in enumerate(frequencies):
        note_file = create_tone(
            duration_seconds=0.5,
            frequency=freq,
            amplitude=0.5,
            output_path=os.path.join(test_dir, f"note_{i+1}.wav")
        )
        scale_files.append(note_file)
    
    print(f"C 메이저 스케일 노트 파일 {len(scale_files)}개 생성 완료")
    
    print("\n모든 테스트가 완료되었습니다.")
    print(f"생성된 파일은 '{test_dir}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 