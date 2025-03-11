#url 이미지를 다운받아 저장합니다. 폴더는 오늘 날짜로 yyyymmdd 폴더에 timestamp_v1 명으로 저장합니다.
#저장한 이미지의 배경을 제거합니다. timestamp_v2 명으로 저장합니다.
#배경을 제거한 이미지를 여백을 제거합니다. timestamp_v3 명으로 저장합니다.
#여백을 제거한 이미지의 전체 색상을 검정색으로 변경합니다. timestamp_v4 명으로 저장합니다.
#여백을 제거한 이미지를 원하는 크기의 정사각형으로 만듭니다. timestamp_v5 명으로 저장합니다.

from Image_Url_DownLoad import download_image
from Image_Remove_Background import remove_white_background_and_vectorize
from Image_Size_Limit import trim_and_square
from Image_Size_Change import resize_image
import os
from datetime import datetime
import time

def process_image(url: str, size: tuple = (1024, 1024)) -> None:
    """
    이미지 처리 파이프라인을 실행합니다.
    
    Args:
        url (str): 다운로드할 이미지 URL
        size (tuple): 최종 이미지 크기 (width, height)
    """
    try:
        # 현재 날짜로 폴더 생성
        today = datetime.now().strftime("%Y%m%d")
        os.makedirs(today, exist_ok=True)
        
        # 타임스탬프 생성
        timestamp = int(time.time())
        
        print(f"현재 날짜: {today}")

        # 1. 이미지 다운로드 (v1)
        v1_path = os.path.join(today, f"{timestamp}_v1.png")
        downloaded_path = download_image(url)
        os.rename(downloaded_path, v1_path)
        
        print(f"v1_path: {v1_path}")

        # 2. 배경 제거 (v2)
        v2_path = os.path.join(today, f"{timestamp}_v2.png")
        remove_white_background_and_vectorize(v1_path, v2_path, threshold=240, blend_size=2, add_border_width=0, make_black=True)
        
        print(f"v2_path: {v2_path}")

        # 3. 여백 제거 (v3)
        v3_path = os.path.join(today, f"{timestamp}_v3.png")
        trim_and_square(v2_path, v3_path, padding=10)
        
        print(f"v3_path: {v3_path}")

        # 4. 크기 조정 (v5)
        v4_path = os.path.join(today, f"{timestamp}_v4.png")
        resize_image(v3_path, v4_path, size=size)
        
        print(f"이미지 처리가 완료되었습니다. 최종 파일: {v4_path}")    
        
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    # 사용 예시
    image_url = input("이미지 URL을 입력하세요: ")
    process_image(image_url)



