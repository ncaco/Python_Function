import torch
import platform
import os
import subprocess
import sys

def get_gpu_info():
    """
    GPU 사용 가능 여부와 정보를 확인하고 결과를 반환합니다.
    
    Returns:
        dict: GPU 정보를 담은 딕셔너리
            - available (bool): GPU 사용 가능 여부
            - device (str): 사용 가능한 디바이스 ('cuda' 또는 'cpu')
            - count (int): 사용 가능한 GPU 개수
            - name (str): GPU 이름 (사용 가능한 경우)
            - memory (float): GPU 메모리 (GB 단위, 사용 가능한 경우)
            - cuda_version (str): CUDA 버전 (사용 가능한 경우)
            - platform (str): 운영체제 정보
    """
    result = {
        "available": False,
        "device": "cpu",
        "count": 0,
        "name": "N/A",
        "memory": 0.0,
        "cuda_version": "N/A",
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})"
    }
    
    # PyTorch를 통한 CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        result["available"] = True
        result["device"] = "cuda"
        result["count"] = torch.cuda.device_count()
        result["cuda_version"] = torch.version.cuda
        
        # GPU 이름 및 메모리 정보 가져오기
        try:
            result["name"] = torch.cuda.get_device_name(0)
            result["memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB 단위로 변환
        except Exception as e:
            result["name"] = f"확인 실패: {str(e)}"
    
    return result

def print_gpu_info():
    """
    GPU 정보를 콘솔에 출력합니다.
    """
    info = get_gpu_info()
    
    print("\n===== GPU 정보 =====")
    print(f"GPU 사용 가능: {'예' if info['available'] else '아니오'}")
    print(f"사용 디바이스: {info['device']}")
    
    if info['available']:
        print(f"GPU 개수: {info['count']}")
        print(f"GPU 이름: {info['name']}")
        print(f"GPU 메모리: {info['memory']:.2f} GB")
        print(f"CUDA 버전: {info['cuda_version']}")
    
    print(f"운영체제: {info['platform']}")
    print("===================\n")
    
    return info

def check_nvidia_smi():
    """
    nvidia-smi 명령어를 실행하여 GPU 상태를 확인합니다.
    
    Returns:
        bool: nvidia-smi 명령어 실행 성공 여부
    """
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n===== NVIDIA-SMI 출력 =====")
            print(result.stdout)
            print("==========================\n")
            return True
        else:
            print("nvidia-smi 명령어 실행 실패")
            return False
    except Exception as e:
        print(f"nvidia-smi 명령어 실행 중 오류 발생: {str(e)}")
        return False

def is_gpu_available():
    """
    GPU 사용 가능 여부만 간단히 확인합니다.
    
    Returns:
        bool: GPU 사용 가능 여부
    """
    return torch.cuda.is_available()

def diagnose_gpu_issues():
    """
    GPU 사용이 안 되는 원인을 진단합니다.
    
    Returns:
        dict: 진단 결과
    """
    diagnosis = {
        "gpu_detected": False,
        "nvidia_driver_installed": False,
        "cuda_installed": False,
        "pytorch_cuda_available": False,
        "issues": [],
        "solutions": []
    }
    
    print("\n===== GPU 문제 진단 =====")
    
    # 1. NVIDIA 드라이버 확인
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        
        if result.returncode == 0:
            diagnosis["nvidia_driver_installed"] = True
            diagnosis["gpu_detected"] = True
            print("✓ NVIDIA 드라이버가 설치되어 있습니다.")
            
            # 드라이버 버전 확인
            driver_version = None
            for line in result.stdout.split('\n'):
                if "Driver Version" in line:
                    driver_version = line.split('Driver Version:')[1].strip().split()[0]
                    break
            
            if driver_version:
                print(f"✓ NVIDIA 드라이버 버전: {driver_version}")
        else:
            print("✗ NVIDIA 드라이버가 설치되어 있지 않거나 실행할 수 없습니다.")
            diagnosis["issues"].append("NVIDIA 드라이버 문제")
            diagnosis["solutions"].append("NVIDIA 웹사이트에서 최신 드라이버를 다운로드하여 설치하세요.")
    except Exception as e:
        print(f"✗ NVIDIA 드라이버 확인 중 오류: {str(e)}")
        diagnosis["issues"].append("NVIDIA 드라이버 문제")
        diagnosis["solutions"].append("NVIDIA 웹사이트에서 최신 드라이버를 다운로드하여 설치하세요.")
    
    # 2. CUDA 설치 확인
    cuda_path = None
    if platform.system() == "Windows":
        for env_var in os.environ:
            if "CUDA_PATH" in env_var:
                cuda_path = os.environ[env_var]
                break
    else:
        cuda_paths = ["/usr/local/cuda", "/usr/local/cuda-11.0", "/usr/local/cuda-11.1", "/usr/local/cuda-11.2", 
                     "/usr/local/cuda-11.3", "/usr/local/cuda-11.4", "/usr/local/cuda-11.5", "/usr/local/cuda-11.6",
                     "/usr/local/cuda-11.7", "/usr/local/cuda-11.8", "/usr/local/cuda-12.0", "/usr/local/cuda-12.1"]
        for path in cuda_paths:
            if os.path.exists(path):
                cuda_path = path
                break
    
    if cuda_path:
        diagnosis["cuda_installed"] = True
        print(f"✓ CUDA가 설치되어 있습니다: {cuda_path}")
        
        # CUDA 버전 확인
        try:
            if platform.system() == "Windows":
                nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, shell=True)
            else:
                nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            
            if nvcc_result.returncode == 0:
                for line in nvcc_result.stdout.split('\n'):
                    if "release" in line.lower():
                        cuda_version = line.split("release")[1].strip().split(",")[0]
                        print(f"✓ CUDA 버전: {cuda_version}")
                        break
        except Exception:
            pass
    else:
        print("✗ CUDA가 설치되어 있지 않거나 환경 변수가 설정되지 않았습니다.")
        diagnosis["issues"].append("CUDA 설치 문제")
        diagnosis["solutions"].append("NVIDIA 웹사이트에서 CUDA Toolkit을 다운로드하여 설치하세요.")
    
    # 3. PyTorch CUDA 지원 확인
    if torch.cuda.is_available():
        diagnosis["pytorch_cuda_available"] = True
        print(f"✓ PyTorch CUDA 사용 가능: {torch.version.cuda}")
    else:
        print("✗ PyTorch에서 CUDA를 사용할 수 없습니다.")
        
        # PyTorch 버전 확인
        print(f"  - PyTorch 버전: {torch.__version__}")
        
        if diagnosis["cuda_installed"]:
            print("  - CUDA는 설치되어 있지만 PyTorch에서 인식하지 못합니다.")
            diagnosis["issues"].append("PyTorch CUDA 호환성 문제")
            diagnosis["solutions"].append("PyTorch를 CUDA 지원 버전으로 재설치하세요: pip install torch --upgrade")
        else:
            diagnosis["issues"].append("PyTorch CUDA 지원 없음")
            diagnosis["solutions"].append("CUDA를 설치한 후 PyTorch를 CUDA 지원 버전으로 재설치하세요.")
    
    # 4. 메모리 사용량 확인
    if diagnosis["nvidia_driver_installed"]:
        try:
            if platform.system() == "Windows":
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], 
                                       capture_output=True, text=True, shell=True)
            else:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], 
                                       capture_output=True, text=True)
            
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    used, total = map(int, line.split(','))
                    used_gb = used / 1024
                    total_gb = total / 1024
                    print(f"✓ GPU {i} 메모리: {used_gb:.2f} GB / {total_gb:.2f} GB 사용 중")
                    
                    if used_gb > total_gb * 0.9:
                        diagnosis["issues"].append(f"GPU {i} 메모리 부족")
                        diagnosis["solutions"].append(f"GPU {i}의 메모리 사용량이 높습니다. 다른 프로그램을 종료하거나 작은 모델을 사용하세요.")
        except Exception:
            pass
    
    # 진단 결과 요약
    print("\n===== 진단 결과 =====")
    if not diagnosis["issues"]:
        print("✓ 모든 검사를 통과했습니다. GPU를 사용할 수 있어야 합니다.")
        print("  그래도 문제가 있다면 다음을 확인해보세요:")
        print("  1. 모델이 GPU를 사용하도록 .to('cuda') 메서드가 호출되었는지 확인")
        print("  2. 다른 프로그램이 GPU를 점유하고 있는지 확인")
        print("  3. PyTorch를 재설치: pip install torch --upgrade")
    else:
        print(f"✗ {len(diagnosis['issues'])}개의 문제가 발견되었습니다:")
        for i, (issue, solution) in enumerate(zip(diagnosis["issues"], diagnosis["solutions"])):
            print(f"  {i+1}. 문제: {issue}")
            print(f"     해결: {solution}")
    
    print("=====================\n")
    return diagnosis

if __name__ == "__main__":
    # GPU 정보 출력
    info = print_gpu_info()
    
    # GPU 사용 가능 여부에 따라 다른 동작 수행
    if info["available"]:
        # nvidia-smi 명령어 실행
        check_nvidia_smi()
        
        # 간단한 CUDA 테스트 실행
        print("\n===== CUDA 테스트 =====")
        try:
            # 간단한 텐서 생성 및 GPU로 이동
            x = torch.rand(10, 10).cuda()
            y = torch.rand(10, 10).cuda()
            z = x @ y  # 행렬 곱
            print(f"✓ CUDA 텐서 연산 성공: {z.shape}")
            print("=====================\n")
        except Exception as e:
            print(f"✗ CUDA 텐서 연산 실패: {str(e)}")
            print("=====================\n")
            # 문제 진단 실행
            diagnose_gpu_issues()
    else:
        # GPU를 사용할 수 없는 경우 문제 진단
        diagnose_gpu_issues()
    
    # 종료 코드 설정 (GPU 사용 가능 여부에 따라)
    sys.exit(0 if info["available"] else 1)
