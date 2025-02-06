from PIL import Image

def trim_and_square(input_path: str, output_path: str = None, padding: int = 10) -> None:
    """
    이미지의 투명한 여백을 제거하고 정사각형으로 만드는 함수
    
    Args:
        input_path (str): 입력 이미지 경로
        output_path (str, optional): 출력 이미지 경로. None일 경우 input_path에 '_trimmed_square'를 추가하여 저장
        padding (int, optional): 추가할 여백 크기 (픽셀). 기본값 10
    """
    # 이미지 열기
    img = Image.open(input_path)
    
    # RGBA가 아닌 경우 RGBA로 변환
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # 알파 채널 가져오기
    alpha = img.getchannel('A')
    
    # 투명하지 않은 영역의 경계 찾기
    bbox = alpha.getbbox()
    
    if bbox:
        # 투명 영역 제거하여 크롭
        img = img.crop(bbox)
        
        # 크롭된 이미지 크기 (패딩 추가)
        width = img.size[0] + (padding * 2)
        height = img.size[1] + (padding * 2)
        
        # 정사각형 크기 결정 (긴 쪽 기준)
        square_size = max(width, height)
        
        # 새 정사각형 이미지 생성 (투명 배경)
        result = Image.new('RGBA', (square_size, square_size), (0, 0, 0, 0))
        
        # 중앙 정렬 위치 계산 (패딩 포함)
        x = (square_size - img.size[0]) // 2
        y = (square_size - img.size[1]) // 2
        
        # 크롭된 이미지 붙여넣기
        result.paste(img, (x, y))
        
        # 저장 경로 설정
        if output_path is None:
            output_path = input_path.rsplit('.', 1)[0] + '_trimmed_square.' + input_path.rsplit('.', 1)[1]
        
        # 결과 저장
        result.save(output_path)

if __name__ == "__main__":
    # 사용 예시
    trim_and_square("test.png", "square.png", padding=10)
