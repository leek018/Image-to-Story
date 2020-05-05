import nltk # conda install -c anaconda nltk
nltk.download('punkt')
from collections import Counter
import pickle
import os
from AI.config import get_config
from AI.preprocess.Caption_preprocess import *
import nltk
import pickle

class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_voca(train_data,threshold,root_path):

    voca = Vocab()
    voca.add_word('<pad>')
    voca.add_word('<start>')
    voca.add_word('<end>')
    voca.add_word('<unk>')

    counter = Counter()
    for key,value in train_data.items():
        for sentence in value:
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            counter.update(tokens)

    words = [word for word,cnt in counter.items() if cnt >= threshold]

    for word in words :
        voca.add_word(word)

    save_path = root_path+'voca_'+str(threshold)+'.pickle'
    with open(save_path,'wb') as f:
        pickle.dump(voca,f)
    return voca

train_data = "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\datasets\\caption\\train_data.json"
threshold = 30
# build_voca(train_data, threshold, saved_path=None)

def load_voca(saved_path):
    with open(saved_path, 'rb') as f:
        voca = pickle.load(f)
        return voca

# training data.json을 로드한 후 캡션을 읽어서 일단 토큰화한다
def tokenized_data(json_path,vocab,type="train"):
    data = {}
    if type is "train":
        data = get_train_data(json_path)
    elif type is "val":
        data = get_val_data(json_path)
    else:
        data = get_test_data(json_path)
    image_list = []
    caption_list = []
    max_len = 0
    for image,captions in data.items():
        image_list.append(image)
        for caption in captions:
           tokens = nltk.tokenize.word_tokenize(caption.lower())
           temp = []
           temp.append(vocab('<start>'))
           temp.extend([vocab(token) for token in tokens])
           temp.append(vocab('<end>'))
           caption_list.append(temp)
    dataset = {}
    dataset['image_list'] = image_list
    dataset['caption_list'] = caption_list
    return dataset

def save_tokenized_data(dataset,AI_DIREC,save_path=None,type="train"):
    if type is "test":
        name = "wordVector_test"
    elif type is "val":
        name = "wordVector_val"
    else:
        name = "wordVector_train_5"
    name += ".pickle"
    if save_path is None:
        save_path = AI_DIREC + get_config()['caption_train_word_saved_path']
    save_path += name
    with open(save_path,'wb') as f:
        pickle.dump(dataset,f)

def load_tokenized_data(load_path):
    data = {}
    with open(load_path,'rb') as file:
        data = pickle.load(file)
    return data

