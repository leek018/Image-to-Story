import os
import shutil
import json
import pandas as pd
from AI.config import get_config
import numpy as np


# Req. 3-1	이미지 경로 및 캡션 불러오기
def get_path_caption(path):
    raw_data = pd.read_csv(path)['image_name| comment_number| comment']
    desc = dict()
    for line in raw_data:
        splited = line.split("|")
        image , caption = splited[0],splited[2]
        if image not in desc :
            desc[image] = list()
        desc[image].append(caption)
    return desc

def get_imagepath_caption(path):
    raw_data = pd.read_csv(path)['image_name| comment_number| comment']
    images, captions = [], []
    for idx, line in enumerate(raw_data):
        splited = line.split("|")
        image, caption = splited[0], splited[2]
        if idx % 5 == 0:
            images.append(image)
        captions.append(caption)
    return images, captions



# Req. 3-2	전체 데이터셋을 분리해 저장하기
def dataset_split_save(path):
    data = get_path_caption(path)
    data_keys = list(data.keys())

    total_size = len(data)
    cfg = get_config()
    src = cfg['image_path']
    dst_train = src+'train\\'
    # total에서 6:2:2로 train ,validation, function_test data를 나눌 것입니다.
    print(src)
    print(dst_train)
    train_ratio = 0.6
    train_size = int(total_size*train_ratio)
    train_data_keys = data_keys[:train_size]
    train_data = dict()
    for image_path in train_data_keys:
        shutil.move(src+image_path,dst_train+image_path)
        train_data[image_path] = data[image_path]

    dst_val = src+'val\\'
    val_ratio = 0.2
    val_size = int(total_size*val_ratio)
    val_data_keys = data_keys[train_size:train_size+val_size]
    val_data = dict()
    for image_path in val_data_keys:
        shutil.move(src+image_path,dst_val+image_path)
        val_data[image_path] = data[image_path]

    dst_test = src+'function_test\\'
    test_data_keys = data_keys[train_size+val_size:]
    test_data = dict()
    for image_path in test_data_keys:
        shutil.move(src+image_path,dst_test+image_path)
        test_data[image_path] = data[image_path]

    with open("../datasets/train_data.json", 'w', encoding='utf-8') as train_json:
        json.dump(train_data, train_json)

    with open("../datasets/val_data.json", 'w', encoding='utf-8') as val_json:
        json.dump(val_data, val_json)

    with open("../datasets/test_data.json", 'w', encoding='utf-8') as test_json:
        json.dump(test_data, test_json)

    return train_data,val_data,test_data


def all_datasets_split(path):
    images, captions = get_imagepath_caption(path)
    images_size = len(images)
    cfg = get_config()
    src = cfg['origin']
    dst_train = src + 'train\\'
    train_ratio = 0.6
    train_size = int(images_size * train_ratio)
    train_data_images = images[:train_size]
    train_data = {}
    for idx, image in enumerate(train_data_images):
        shutil.move(src + image, dst_train + image)
        train_data[image] = captions[idx:idx+5]
    dst_val = src + 'validation\\'
    val_ratio = 0.2
    val_size = int(images_size * val_ratio)
    val_data_keys = images[train_size:train_size + val_size]
    val_data = dict()
    for idx, image_path in enumerate(val_data_keys):
        shutil.move(src + image_path, dst_val + image_path)
        val_data[image_path] = captions[train_size + idx:train_size + idx+5]

    dst_test = src + 'function_test\\'
    test_data_keys = images[train_size + val_size:]
    test_data = dict()
    for idx, image_path in enumerate(test_data_keys):
        shutil.move(src + image_path, dst_test + image_path)
        test_data[image_path] = captions[train_size + val_size + idx : train_size + val_size + idx + 5]

    with open("../datasets/train_data_yyejej.json", 'w', encoding='utf-8') as train_json:
        json.dump(train_data, train_json)

    with open("../datasets/val_data_yyejej.json", 'w', encoding='utf-8') as val_json:
        json.dump(val_data, val_json)

    with open("../datasets/test_data_yyejej.json", 'w', encoding='utf-8') as test_json:
        json.dump(test_data, test_json)

    return train_data, val_data, test_data

# a = "../datasets/captions.csv"
# all_datasets_split(a)

# Req. 3-3	저장된 데이터셋 불러오기
def get_train_data(path):
    train_data = {}
    with open(path,'r') as f:
        train_data = json.load(f)
    return train_data


def get_test_data(path):
    test_data = {}
    with open(path, 'r') as f:
        test_data = json.load(f)
    return test_data

def get_val_data(path):
    val_data = {}
    with open(path, 'r') as f:
        val_data = json.load(f)
    return val_data

# Req. 3-4	데이터 샘플링
def sampling_data(ratio,data):
    if ratio >1  or ratio <=0:
        print("0 < ratio < 1")
        return
    sample_data = dict()
    size = len(data)
    target_size = int(ratio * size)
    # 얕은 복사 data가 사라지면 sample_data의 요소도 사라짐
    count = 0
    for key , value in data.items():
        if count > target_size:
            break
        count+=1
        sample_data[key] = data[key]
    return sample_data
