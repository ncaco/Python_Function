from PIL import Image

def resize_image(input_path, output_path, size=(1024, 1024)):
    try:
        # 이미지 열기
        img = Image.open(input_path)
        
        # 이미지 리사이즈
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        
        # 저장
        resized_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"이미지 리사이즈 중 오류 발생: {str(e)}")
        return False

# 실행
input_file = "은하수 도장.png"
output_file = "은하수 도장.png"
resize_image(input_file, output_file)
