from PIL import Image
import requests
import numpy as np
import os
from datetime import datetime
import io

def optimize_image_pipeline(url: str, size: tuple = (1024, 1024)) -> str:
    """
    이미지 URL을 받아 최적화된 이미지 처리 파이프라인 실행
    
    Args:
        url (str): 이미지 URL
        size (tuple): 최종 출력 크기
    Returns:
        str: 저장된 파일 경로
    """
    try:
        # 이미지 다운로드 & 메모리에 로드
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert('RGBA')
        
        # 배경 제거 & 검정색 변환
        img_array = np.array(img)
        mask = np.all(img_array[:, :, :3] >= 240, axis=2)
        img_array[mask, 3] = 0
        img_array[~mask, :3] = 0  # 검정색 변환
        
        img = Image.fromarray(img_array)
        
        # 여백 제거
        bbox = img.getchannel('A').getbbox()
        if bbox:
            img = img.crop(bbox)
            
            # 정사각형 변환
            max_size = max(img.size)
            square = Image.new('RGBA', (max_size, max_size), (0, 0, 0, 0))
            paste_x = (max_size - img.size[0]) // 2
            paste_y = (max_size - img.size[1]) // 2
            square.paste(img, (paste_x, paste_y))
            
            # 최종 크기 조정
            final = square.resize(size, Image.Resampling.LANCZOS)
            
            # 저장
            today = datetime.now().strftime("%Y%m%d")
            os.makedirs(today, exist_ok=True)
            output_path = os.path.join(today, f"{int(datetime.now().timestamp())}.png")
            final.save(output_path, "PNG")
            
            return output_path
            
    except Exception as e:
        print(f"이미지 처리 실패: {str(e)}")
        return None

if __name__ == "__main__":
    url = input("이미지 URL을 입력하세요: ")
    result = optimize_image_pipeline(url)
    if result:
        print(f"처리 완료: {result}")