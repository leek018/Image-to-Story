import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ImageCaptionDataset(Dataset):
    def __init__(self,image_path,image_data,caption_data,transform):
        self.root = image_path
        self.caption_data = caption_data
        self.image_data = image_data
        self.transform = transform
    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, caption_index):
        #get image
        image_index = caption_index // 5
        image = Image.open(self.root+self.image_data[image_index])
        image = self.transform(image)
        vector_caption = torch.LongTensor(self.caption_data[caption_index])
        return image,vector_caption



def make_caption_loader(Image_caption_dataset, batch, image_path):
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),  # 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4444, 0.4215, 0.3833), std=(0.2738, 0.2664, 0.2766))  # get from req1
    ])

    dataset = ImageCaptionDataset(image_path= image_path,image_data=Image_caption_dataset['image_list'],caption_data=Image_caption_dataset['caption_list'],transform=transform)
    loader = DataLoader(dataset,batch_size=batch,shuffle=True,collate_fn=collate_fn)

    return loader

def collate_fn(data):
    # req3 leek_ImageCaptionDatset.py 에 구현된
    # data는 ImageCaptionDataset의__getitem__으로부터 batch사이즈 만큼 데이터를 callback 으로 받음.
    # data type은 list이고 element는 tuple(image,caption)
    # 받은 데이터를 TIL에 적은대로 구현한다.

    # caption 길이를 기준으로 sort한다.
    data.sort(key=lambda element : len(element[1]),reverse=True)

    # 이후 image는 image대로 caption은 caption대로 모아줘서 리턴해줘야 함으로 분리해준다.
    images, captions = zip(*data)

    # 패딩 되기 이전의 실제 caption의 길이를 기록해야 한다.
    # 이때, batch 사이즈에서 가장 길이가 큰 값을 기준으로 모자란 부분을 패딩 해줘야함
    # 참고로 이후 사용할 pack_padded_sequence 함수를 위해서 길이를 기록해놔야 함
    # 이에 대해서는 TIL 2020 04 02에 기록하였음
    caption_length = [len(caption) for caption in captions ]

    # 2차원짜리 data형[batchsize X 가장 길이가 긴 캡션의 길이 ]을 만들어 놔야한다.
    # zeros로 선언한 이유는 0을 패딩으로 설정 했기 때문.
    padded_caption = torch.zeros(size=(len(caption_length),max(caption_length))).long()
    # caption_length를 temp에 매핑시킨다.
    for index, caption in enumerate(captions):
        padded_caption[index, 0:caption_length[index]] = caption[ 0:caption_length[index]]

    # 현재 image는 list 타입으로 10개의 3차원 짜리 이미지들이 1D으로 구성된 상황
    # ex) [ (3,224,224) , (3,224,224), (3,224,224)....]
    # 이걸 ( 10, 3,224,224)의 torch 데이터 타입으로 바꿀 것이다.
    images = torch.stack(images,dim=0)

    return images,padded_caption,caption_length

