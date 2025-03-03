import subprocess
import sys
import os

def install_requirements():
    """필요한 패키지를 설치하는 함수"""
    requirements = [
        "torch",
        "librosa",
        "numpy",
        "transformers",
        "tkinter",
    ]
    
    print("필요한 패키지 설치 중...")
    
    for package in requirements:
        try:
            if package == "tkinter":
                # tkinter는 pip로 설치할 수 없으므로 건너뜁니다.
                continue
                
            print(f"{package} 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} 설치 완료")
        except Exception as e:
            print(f"{package} 설치 중 오류 발생: {str(e)}")
            
    print("\n모든 패키지 설치 완료")
    print("\n주의: tkinter는 Python과 함께 설치되어야 합니다.")
    print("Windows에서는 Python 설치 시 'tcl/tk and IDLE' 옵션을 선택해야 합니다.")
    print("Linux에서는 'sudo apt-get install python3-tk'와 같은 명령으로 설치할 수 있습니다.")

if __name__ == "__main__":
    install_requirements()
    
    print("\n설치가 완료되었습니다. 이제 sttModel.py를 실행할 수 있습니다.")
    print("사용 예시:")
    print("python sttModel.py")
    print("python sttModel.py --model openai/whisper-tiny")
    print("python sttModel.py --file path/to/audio.wav") 