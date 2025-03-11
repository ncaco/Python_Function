import requests
import os
from urllib.parse import urlparse

def download_image(url: str) -> str:
    """
    URL에서 이미지를 다운로드하여 현재 디렉토리에 저장합니다.
    
    Args:
        url (str): 다운로드할 이미지 URL
    
    Returns:
        str: 저장된 이미지 파일 경로
    """
    try:
        # URL에서 이미지 다운로드
        response = requests.get(url)
        response.raise_for_status()
        
        # URL에서 파일명 추출
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = 'image.jpg'
            
        # 현재 디렉토리에 파일 저장
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        return filepath
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"이미지 다운로드 실패: {str(e)}")

# 사용 예시
if __name__ == "__main__":
    image_url = "https://thumbs.dreamstime.com/b/fire-flame-icon-isolated-bonfire-sign-emoticon-symbol-white-emoji-logo-illustration-vector-142833014.jpg"
    try:
        saved_path = download_image(image_url)
        print(f"이미지가 성공적으로 저장되었습니다: {saved_path}")
    except Exception as e:
        print(f"에러 발생: {str(e)}")
