import os
from datetime import datetime
from pytz import timezone

def create_dir(directory):
    # 주어진 경로가 존재하지 않을 경우 디렉토리를 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
# pretrain/sasrec/pretrain_data 안에 있는 파일 경로 저장하기 (리스트)
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path): # target_path에 있는 파일 및 디렉토리 이름 리스트 반환
        if os.path.isfile(os.path.join(target_path, file)): # 파일이라면
            if target_word in file: # target_word(.pth)가 파일 이름에 포함되어 있다면
                file_paths.append(target_path + file) # 파일 경로를 리스트에 추가
            
    return file_paths

    
    
    