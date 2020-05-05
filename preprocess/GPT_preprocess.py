# -*- coding:utf-8 -*-
from nltk.tokenize import sent_tokenize
import re
import pickle

# 1. 문장 토큰화
def sent_tokenizor(read, write):
    f = open(read, 'r', encoding='UTF8') # 읽기
    ff = open(write, 'w', -1, 'utf-8') # 쓰기

    for s in f:
        ff.write('\n')
        for sent in sent_tokenize(s):
            ff.write('\n'+sent)

    f.close()
    ff.close()

# sent_tokenizor('life_data.txt', 'life_split_data.txt')
# sent_tokenizor('story_data.txt', 'story_split_data.txt')


# 2. 데이터 전처리
def preprocess(text):
    text = re.sub(pattern='Posted on [0-9]{4} [0-9]{2} [0-9]{2} .+ posted in \S+ \s?', repl='', string=text)
    text = re.sub(pattern='Posted on [0-9]{8} .+ posted in \S+ \s?', repl='', string=text)
    text = re.sub(pattern='[0-9]{4}년 [0-9]{,2}월 [0-9]{,2}일 [0-9]{,2}시 [0-9]{,2}분 [0-9]{,2}초', repl='', string=text)
    text = re.sub(pattern='[0-9]{4}. [0-9]{,2}. [0-9]{,2}', repl='', string=text)
    _filter = re.compile('[ㄱ_|]+')
    text = _filter.sub('', text)
    _filter = re.compile('[^가-힣 0-9 a-z A-Z \. \, \' \" \? \!]+')
    text = _filter.sub('', text)
    return text


# 수필
# f = open('life_split_data.txt', 'r', -1, "utf-8") # 읽기
# ff = open('life_preprocess_data.txt', 'w', -1, 'utf-8')  # 쓰기
#
# for s in f:
#     if len(preprocess(s)) >= 3 and '.' in s or '?' in s or '!' in s:
#         ff.write('\n' + preprocess(s))
#
# f.close()
# ff.close()


# 소설
# f = open('story_split_data.txt', 'r', -1, "utf-8")
# ff = open('story_preprocess_data.txt', 'w', -1, 'utf-8')
#
# for s in f:
#     if len(preprocess(s)) >= 3 and '.' in s or '?' in s or '!' in s:
#         ff.write('\n' + preprocess(s))
#
# f.close()
# ff.close()


# 3. 데이터 분리, 피클 저장
def savePickle(name, data):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(data, f)


def splitData(read, name):
    f = open(read, 'r', -1, "utf-8")
    data_list = []
    for s in f:
        if '\n' in s:
            data_list.append(s.replace('\n', ''))
        else:
            data_list.append(s)
    print(data_list)
    L = len(data_list)
    data_dic = {}
    train_dic = {}
    test_dic = {}
    data_dic[name] = data_list
    train_dic[name] = data_list[:int(L*0.8)]
    test_dic[name] = data_list[int(L*0.8):]

    savePickle(name+'_pkl', data_dic)
    savePickle(name+'_train_pkl', train_dic)
    savePickle(name+'_test_pkl', test_dic)

# splitData('life_preprocess_data.txt', 'life')
splitData('story_preprocess_data.txt', 'story')