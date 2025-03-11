# 파일 탐색기로 파일을 선택하여 해당 파일의 경로를 반환합니다.
# 추가기능
# 1. 파라미터로 확장자 하나를 입력하면 해당 확장자만 탐색합니다.
# 2. 파라미터로 확장자 여러개를 입력하면 해당 확장자들만 탐색합니다.
# 3. 제외할 확장자를 하나 입력하면 해당 확장자는 탐색하지 않습니다.
# 4. 제외할 확장자 여러개를 입력하면 해당 확장자들은 탐색하지 않습니다.

import tkinter as tk
from tkinter import filedialog
import os

def _normalize_extension(ext):
    """확장자 형식을 정규화합니다 (.을 포함한 형태로 변환)"""
    if not ext:
        return None
    return ext if ext.startswith('.') else f'.{ext}'

def _normalize_extensions(extensions):
    """확장자 목록을 정규화합니다"""
    if not extensions:
        return []
    
    # 단일 확장자인 경우 리스트로 변환
    if isinstance(extensions, str):
        return [_normalize_extension(extensions)]
    # 리스트나 튜플인 경우 각 항목을 정규화
    elif isinstance(extensions, (list, tuple)):
        return [_normalize_extension(ext) for ext in extensions if ext]
    
    return []

def select_file(extension=None, exclude_extension=None, title="파일 선택", initial_dir=None):
    """
    파일 탐색기로 파일을 선택하여 해당 파일의 경로를 반환합니다.
    
    Args:
        extension (str, list, tuple): 허용할 확장자 또는 확장자 목록
        exclude_extension (str, list, tuple): 제외할 확장자 또는 확장자 목록
        title (str): 파일 탐색기 창의 제목
        initial_dir (str): 초기 디렉토리 경로
        
    Returns:
        str or None: 선택한 파일 경로 또는 취소/제외된 경우 None
    """
    try:
        root = tk.Tk()
        root.withdraw()
        
        # 초기 디렉토리 설정
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()
        
        # 확장자 정규화
        allowed_extensions = _normalize_extensions(extension)
        excluded_extensions = _normalize_extensions(exclude_extension)
        
        # 파일 유형 필터 설정
        filetypes = []
        
        # 허용 확장자 처리
        if allowed_extensions:
            for ext in allowed_extensions:
                ext_name = ext[1:].upper()  # .을 제외한 확장자 이름
                filetypes.append((f'{ext_name} 파일', f'*{ext}'))
        
        # 필터가 없으면 모든 파일 표시
        if not filetypes:
            filetypes.append(('모든 파일', '*.*'))
        
        # 파일 선택 대화상자 표시
        file_path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        # 사용자가 취소한 경우
        if not file_path:
            return None
        
        # 제외할 확장자 처리
        if excluded_extensions:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [ext.lower() for ext in excluded_extensions]:
                return None
        
        return file_path
        
    except Exception as e:
        print(f"파일 선택 중 오류 발생: {e}")
        return None
    finally:
        # 메인 루프가 실행 중이면 종료
        try:
            root.destroy()
        except:
            pass


def select_files(extension=None, exclude_extension=None, title="파일 선택", initial_dir=None):
    """
    파일 탐색기로 여러 파일을 선택하여 해당 파일들의 경로 리스트를 반환합니다.
    
    Args:
        extension (str, list, tuple): 허용할 확장자 또는 확장자 목록
        exclude_extension (str, list, tuple): 제외할 확장자 또는 확장자 목록
        title (str): 파일 탐색기 창의 제목
        initial_dir (str): 초기 디렉토리 경로
        
    Returns:
        list or None: 선택한 파일 경로 리스트 또는 취소/제외된 경우 빈 리스트
    """
    try:
        root = tk.Tk()
        root.withdraw()
        
        # 초기 디렉토리 설정
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()
        
        # 확장자 정규화
        allowed_extensions = _normalize_extensions(extension)
        excluded_extensions = _normalize_extensions(exclude_extension)
        
        # 파일 유형 필터 설정
        filetypes = []
        
        # 허용 확장자 처리
        if allowed_extensions:
            for ext in allowed_extensions:
                ext_name = ext[1:].upper()  # .을 제외한 확장자 이름
                filetypes.append((f'{ext_name} 파일', f'*{ext}'))
        
        # 필터가 없으면 모든 파일 표시
        if not filetypes:
            filetypes.append(('모든 파일', '*.*'))
        
        # 파일 선택 대화상자 표시
        file_paths = filedialog.askopenfilenames(
            title=title,
            initialdir=initial_dir,
            filetypes=filetypes
        )
        
        # 사용자가 취소한 경우
        if not file_paths:
            return []
        
        # 제외할 확장자 처리
        if excluded_extensions:
            excluded_exts_lower = [ext.lower() for ext in excluded_extensions]
            filtered_paths = []
            
            for path in file_paths:
                file_ext = os.path.splitext(path)[1].lower()
                if file_ext not in excluded_exts_lower:
                    filtered_paths.append(path)
            
            return filtered_paths
        
        return list(file_paths)
        
    except Exception as e:
        print(f"파일 선택 중 오류 발생: {e}")
        return []
    finally:
        # 메인 루프가 실행 중이면 종료
        try:
            root.destroy()
        except:
            pass


def select_directory(title="디렉토리 선택", initial_dir=None):
    """
    파일 탐색기로 디렉토리를 선택하여 해당 디렉토리 경로를 반환합니다.
    
    Args:
        title (str): 파일 탐색기 창의 제목
        initial_dir (str): 초기 디렉토리 경로
        
    Returns:
        str or None: 선택한 디렉토리 경로 또는 취소된 경우 None
    """
    try:
        root = tk.Tk()
        root.withdraw()
        
        # 초기 디렉토리 설정
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.getcwd()
        
        # 디렉토리 선택 대화상자 표시
        dir_path = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir
        )
        
        return dir_path if dir_path else None
        
    except Exception as e:
        print(f"디렉토리 선택 중 오류 발생: {e}")
        return None
    finally:
        # 메인 루프가 실행 중이면 종료
        try:
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    # 테스트 코드
    print("단일 파일 선택:", select_file())
    print("특정 확장자 파일 선택:", select_file(extension=[".mp3", ".wav"], title="음악 파일 선택"))
    print("특정 확장자 제외 파일 선택:", select_file(exclude_extension=".mp3", title="MP3 제외 파일 선택"))
    print("여러 파일 선택:", select_files(extension=".txt", title="텍스트 파일 선택"))
    print("디렉토리 선택:", select_directory())
