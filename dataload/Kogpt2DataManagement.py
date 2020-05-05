import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def build_korean_to_idx(file_path,save_path,vocab,tokenizer,block_size):
    with open(file_path,'rb') as f:
        raw_data = pickle.load(f)
    #data[0]이 이상한 문자라서 삭제하고 처리함.
    #total_tokend = dict()
    temp = []
    output = []
    save_name = save_path
    output_dict = dict()
    for key,value in raw_data.items():
        value = value[1:]
        for string in value:
            tokened = tokenizer(string)
            temp += vocab[tokened]
        limit = len(temp)//block_size
        for i in range(limit):
            output.append(temp[i*block_size:(i+1)*block_size])
        save_name = save_path + key+"_to_idx.pkl"
        output_dict[key] = output
        print(len(output))
        with open(save_name,"wb") as file:
            pickle.dump(output_dict,file)

class Kogpt2Dataset(Dataset):

    def __init__(self,file_path):
        with open(file_path,'rb') as file:
            pickle_data = pickle.load(file)
        self.data = next(iter(pickle_data.values()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item], dtype=torch.long)

def make_kogpt2_loader(idx_pickle_path,batch):

    dataset = Kogpt2Dataset(idx_pickle_path)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return loader