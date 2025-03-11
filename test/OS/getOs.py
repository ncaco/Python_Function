import platform
import os
import sys

def get_os_info():
    """
    현재 시스템의 OS 정보를 반환하는 함수
    
    Returns:
        dict: 다양한 OS 정보를 포함하는 딕셔너리
    """
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "node": platform.node()
    }
    
    # 운영체제별 추가 정보
    if platform.system() == "Windows":
        os_info["windows_edition"] = platform.win32_edition() if hasattr(platform, "win32_edition") else "Not available"
        os_info["windows_ver"] = platform.win32_ver()
    elif platform.system() == "Linux":
        os_info["linux_distribution"] = platform.freedesktop_os_release() if hasattr(platform, "freedesktop_os_release") else "Not available"
    elif platform.system() == "Darwin":  # macOS
        os_info["mac_ver"] = platform.mac_ver()
    
    return os_info

def print_os_info():
    """
    시스템의 OS 정보를 콘솔에 출력하는 함수
    """
    os_info = get_os_info()
    print("=" * 50)
    print("시스템 OS 정보")
    print("=" * 50)
    
    for key, value in os_info.items():
        print(f"{key}: {value}")
    
    print("=" * 50)

# 이 파일이 직접 실행될 때만 OS 정보 출력
if __name__ == "__main__":
    print_os_info()
