from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
import svgwrite

def add_border(image: Image.Image, border_width: int = 2, border_color: tuple = (0, 0, 0, 255)) -> Image.Image:
    """
    이미지에 테두리를 추가합니다.
    
    Args:
        image: PIL Image 객체
        border_width: 테두리 두께 (기본값: 2px)
        border_color: 테두리 색상 (R,G,B,A) (기본값: 검정색)
    
    Returns:
        테두리가 추가된 이미지
    """
    # 이미지 크기 가져오기
    width, height = image.size
    
    # 새 이미지 생성 (원본 + 테두리)
    new_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(new_image)
    
    # 원본 이미지의 알파 채널을 마스크로 사용
    alpha = image.split()[3]
    mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # 알파 채널이 0이 아닌 픽셀 찾기
    alpha_data = np.array(alpha)
    y_indices, x_indices = np.where(alpha_data > 0)
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        # 외곽선 그리기
        for i in range(-border_width, border_width + 1):
            for j in range(-border_width, border_width + 1):
                if i*i + j*j <= border_width*border_width:
                    mask_draw.point(list(zip(x_indices + i, y_indices + j)), fill=255)
    
    # 테두리 그리기
    new_image.paste(border_color, mask=mask)
    # 원본 이미지 붙이기
    new_image.paste(image, (0, 0), image)
    
    return new_image

def convert_to_black(image: Image.Image) -> Image.Image:
    """
    이미지의 모든 색상을 검정색으로 변환합니다.
    알파 채널은 유지됩니다.
    
    Args:
        image: PIL Image 객체
    
    Returns:
        검정색으로 변환된 이미지
    """
    # RGBA 배열로 변환
    img_array = np.array(image)
    
    # 알파 채널이 0이 아닌 픽셀만 검정색으로 변경
    mask = img_array[:, :, 3] > 0
    img_array[mask, 0:3] = [0, 0, 0]  # RGB를 검정색으로
    
    return Image.fromarray(img_array)

def remove_white_background_and_vectorize(input_path: str, output_path: str = None, threshold: int = 240, 
                                        blend_size: int = 2, vector_output: str = None, 
                                        add_border_width: int = 0, make_black: bool = False) -> tuple:
    """
    흰색 배경을 제거하고 테두리를 블렌딩한 후 벡터 이미지로 변환합니다.
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        threshold: 흰색 판단 임계값 (기본값: 240)
        blend_size: 블렌딩할 테두리 크기 (기본값: 2)
        vector_output: 벡터 이미지 출력 경로 (기본값: None)
        add_border_width: 테두리 두께 (0이면 테두리 미적용, 기본값: 0)
        make_black: 모든 색상을 검정색으로 변경 (기본값: False)
    
    Returns:
        tuple: (png_path, svg_path)
    """
    # 이미지 로드 및 RGBA 변환
    img = Image.open(input_path)
    img = img.convert("RGBA")
    
    # numpy 배열로 변환
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # 알파 마스크 생성 (흰색 픽셀 찾기)
    is_white = np.all(img_array[:, :, :3] >= threshold, axis=2)
    
    # 배경 제거
    img_array[is_white, 3] = 0
    
    # 테두리 블렌딩
    for y in range(blend_size, height-blend_size):
        for x in range(blend_size, width-blend_size):
            if is_white[y, x]:  # 제거된 배경 픽셀인 경우
                # 주변 픽셀 영역 추출
                region = img_array[y-blend_size:y+blend_size+1, x-blend_size:x+blend_size+1]
                region_mask = ~is_white[y-blend_size:y+blend_size+1, x-blend_size:x+blend_size+1]
                
                # 배경이 아닌 픽셀만 선택
                valid_pixels = region[region_mask]
                
                if len(valid_pixels) > 0:
                    # 주변 색상의 평균으로 설정
                    img_array[y, x, :3] = np.mean(valid_pixels[:, :3], axis=0)
    
    # 테두리 추가
    if add_border_width > 0:
        result_img = add_border(Image.fromarray(img_array), border_width=add_border_width)
    else:
        result_img = Image.fromarray(img_array)
    
    # 검정색 변환
    if make_black:
        result_img = convert_to_black(result_img)
    
    # 결과 이미지 생성
    if output_path is None:
        file_name = os.path.splitext(input_path)[0]
        output_path = f"{threshold}_{blend_size}.png"
    
    # 저장
    result_img.save(output_path, "PNG")

    return output_path

if __name__ == "__main__":
    try:
        # 테두리 2px 추가 예제
        png_path = remove_white_background_and_vectorize(
            "img.png", 
            threshold=200,     # 흰색 판단 임계값
            blend_size=3,      # 테두리 블렌딩 크기
            add_border_width=0,# 테두리 두께
            make_black=True    # 모든 색상을 검정색으로 변경
        )
        print(f"배경이 제거된 이미지가 저장되었습니다: {png_path}")
    except Exception as e:
        print(f"오류 발생: {e}")
