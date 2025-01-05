import os
import os.path
import gzip
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

def parse(path):
    g = gzip.open(path, 'rb') # gzip 파일을 읽기 모드('rb')로 엶
    for l in tqdm(g): # gzip 파일의 각 줄을 순차적으로 읽음
        yield json.loads(l) # 읽은 줄을 JSON 형식으로 변환하여 반환 (generator)
        
def preprocess(fname):
    countU = defaultdict(lambda: 0) # user count / key가 존재하지 않으면 기본값 0 반환
    countP = defaultdict(lambda: 0) # item count
    line = 0 # line number

    file_path = f'../../data/amazon/{fname}.json.gz' 
    
    # counting interactions for each user and item
    for l in parse(file_path):
        line += 1
        asin = l['asin'] # item id
        rev = l['reviewerID'] # user id
        time = l['unixReviewTime'] # timestamp
        countU[rev] += 1 # count interactions for each user
        countP[asin] += 1 # count interactions for each item
    
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    review_dict = {}
    name_dict = {'title':{}, 'description':{}}
    
    f = open(f'../../data/amazon/meta_{fname}.json', 'r')
    json_data = f.readlines()
    f.close()
    data_list = [json.loads(line[:-1]) for line in json_data] # meta data 전처리 후 list로 저장
    meta_dict = {} # key: item id, value: meta data
    for l in data_list:
        meta_dict[l['asin']] = l
    
    for l in parse(file_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        
        threshold = 5
        if ('Beauty' in fname) or ('Toys' in fname):
            threshold = 4 # Beauty, Toys & Games 데이터셋은 threshold를 4로 설정
            
        if countU[rev] < threshold or countP[asin] < threshold: # 논문에서 설명한 것처럼 threshold보다 작은 user/item interaction은 제외
            continue
        
        if rev in usermap: # user id가 usermap에 존재하면
            userid = usermap[rev] # user id를 usermap에서 가져옴. 
        else:
            usernum += 1 
            userid = usernum 
            usermap[rev] = userid # user id를 usermap에 추가
            User[userid] = [] # user id를 key로 하는 빈 list 생성
        
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid]) # user id에 대한 interaction 추가 (sequence)
        
        
        if itemmap[asin] in review_dict: 
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                a = 0
        else:
            review_dict[itemmap[asin]] = {'review': {}, 'summary':{}} # item id에 대한 review 추가
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText'] # user id에 대한 review 추가
            except:
                a = 0 # review가 없는 경우
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary'] # user id에 대한 summary 추가
            except:
                a = 0 # summary가 없는 경우
        try:
            if len(meta_dict[asin]['description']) ==0: # description이 없는 경우
                name_dict['description'][itemmap[asin]] = 'Empty description' # 'Empty description'으로 설정
            else:
                name_dict['description'][itemmap[asin]] = meta_dict[asin]['description'][0] # description 추가
            name_dict['title'][itemmap[asin]] = meta_dict[asin]['title'] # title 추가
        except:
            a =0 
    
    with open(f'../../data/amazon/{fname}_text_name_dict.json.gz', 'wb') as tf: # item id에 대한 title, description 저장
        pickle.dump(name_dict, tf) # title, description 저장
    
    for userid in User.keys(): # user id에 대한 interaction을 시간 순으로 정렬
        User[userid].sort(key=lambda x: x[0])
        
    print(usernum, itemnum) # user 수, item 수 출력
    
    f = open(f'../../data/amazon/{fname}.txt', 'w') # user id, item id 저장
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()
    
    
# if __name__ == '__main__':
#     preprocess('All_Beauty')
    
# 최종적으로 저장된 정보
# 1. user id, item id 저장된 파일: All_Beauty.txt
# 2. item id에 대한 title, description 저장된 파일: All_Beauty_text_name_dict.json.gz
