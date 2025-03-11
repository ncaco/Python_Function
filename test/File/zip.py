import zipfile

# ZIP 파일 열기
with zipfile.ZipFile('../Audio/Audio.zip', 'r') as zip_ref:
    # 모든 내용 압축 해제
    zip_ref.extractall('result')
    
    # 특정 파일만 압축 해제
    # zip_ref.extract('특정파일명', '대상경로')