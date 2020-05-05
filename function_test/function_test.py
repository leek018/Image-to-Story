from AI.dataload.CaptionDataManagement import *
from AI.models.Captionmodel.Encoder import EncoderCNN
from AI.models.Captionmodel.Decoder import DecoderRNN
from AI.train.CaptionTrain import caption_train
from AI.train.AttentionTrain import attention_caption_train
from AI.test.CaptionPredict import caption_test
from AI.vocab.CaptionVocab import load_voca,tokenized_data,save_tokenized_data,load_tokenized_data,build_voca
from AI.utils import *
from AI.config import get_config, get_config_yye,get_kog_config
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import datetime
import pickle
#KgGPT
from AI.models.Original_kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from AI.models.Original_kogpt2.utils import get_tokenizer
import re
#OurGPT
from AI.train.FinetuningGPT2 import fine_tuning

#AI SERVICE
from AI.prediction import predict,korean_postprocess
#fine tune
import gluonnlp as nlp
from AI.dataload.Kogpt2DataManagement import build_korean_to_idx,Kogpt2Dataset
from transformers import GPT2Config, GPT2LMHeadModel

#prediction
import copy

#attention
from torchvision import transforms as torch_transform
from AI.models.Captionmodel.models import Encoder,DecoderWithAttention
from prediction import load_image
import torch.nn.functional as F
def build_voca_test():
    config = get_config()
    train_json_path = config['caption_train_path']
    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    voca_root_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\caption\\"
    build_voca(train_data,5,voca_root_path)
def save_caption2idx_test():
    config = get_config()
    AI_DIREC="C:\\Users\\multicampus\\s02p23c104\\Back\\AI"
    voca_path = AI_DIREC + config['caption_vocab_path']
    train_json_path = AI_DIREC + config['caption_train_path']
    voca = load_voca(voca_path)
    dataset = tokenized_data(train_json_path, voca)
    save_tokenized_data(dataset=dataset,AI_DIREC=AI_DIREC)

def loader_test():
    config = get_config()
    load_path = config['word2idx_test_path']
    voca_path = config['caption_vocab_path']
    dataset = load_tokenized_data(load_path)
    print(dataset['image_list'])
    voca = load_voca(voca_path)

    loader = make_caption_loader(dataset, 10, config['train_image_path'])
    dataiter = iter(loader)
    images, padded_caption,caption_length = dataiter.next()
    print(images)

def embed_feature_cat_test():

    batch = 3
    embed_size = 6
    vocab_size = 6
    word_max_size = 5
    feature = torch.randn(size=(batch,embed_size))
    print("feature")
    print(feature)
    embed_layer = nn.Embedding(vocab_size,embed_size)

    test_word2idx = torch.zeros(size=(batch,word_max_size)).long()
    for i in range(batch):
        for j in range(batch-i):
            test_word2idx[i,j] = batch-i+j
    print("test_word2idx")
    print(test_word2idx)

    embeddings = embed_layer(test_word2idx)
    print("embeddings")
    print(embeddings)

    print("embeddigs cat to feature")
    cat_result = torch.cat([feature.unsqueeze(1),embeddings],1)
    print(cat_result)

    test_word2idx_length = [3,2,1]
    pack = pack_padded_sequence(cat_result,test_word2idx_length,batch_first=True)
    print("after pack")
    print(pack.data)

    print("after lstm")
    output = pack_lstm_test(pack,embed_size)
    print(output)

def pack_lstm_test(pack,embed_size):
    hidden_layers = 1
    hidden_size = 10
    lstm = nn.LSTM(embed_size,hidden_size,hidden_layers)

    output,_ = lstm(pack)
    return output.data

def train_procedure_test():
    config = get_config()
    load_path = config['word2idx_train_path']
    voca_path = config['caption_vocab_path']
    dataset = load_tokenized_data(load_path)
    voca = load_voca(voca_path)
    batch_size = 2
    embed_size = 10
    vocab_len = len(voca)
    hidden_layer = 1
    hidden_size = 10
    loader = make_caption_loader(dataset,batch_size,config['caption_train_image_path'])

    dataiter = iter(loader)
    images,caption,length = dataiter.next()

    # data형태 확인하기
    print("Data 형태 확인")
    print(images.size())
    print(caption.size())


    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size,vocab_len,hidden_layer,hidden_size)

    grad_params = list(encoder.linear.parameters())

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=grad_params,lr=0.001)

    compare_target = pack_padded_sequence(caption,length,batch_first=True).data

    feature = encoder(images)
    output = decoder(caption,feature,length)

    loss = loss_function(output,compare_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    datestr = date2str()
    save_path = config['checkpoints_saved_path']
    mini_batch_loss = []
    mini_batch_loss.append(loss.item())
    save_config(config,"config"+datestr,save_path)
    save_loss(mini_batch_loss, "loss" + datestr, save_path)
    save_model(encoder, "encoder" + datestr, save_path)
    save_model(decoder, "decoder" + datestr, save_path)
    print("optimzer.zero_grad()와 encoder.zero_grad() , decoder.zero_grad()와 같을 까?")
    print("optimizer.zero_grad() 호출하기 전")
    print(encoder.linear.weight.grad)
    print("optimizer.zero_grad() 호출한 후")
    optimizer.zero_grad()
    print(encoder.linear.weight.grad)
    print("====================")
    print(grad_params)


def caption_train_test():
    config = get_config()
    print(config)
    vocab_path = config['caption_vocab_path']
    word2idx_path = config['word2idx_train_path']
    image_path = config['caption_train_image_path']
    save_path = config['checkpoints_saved_path']
    caption_path = config['caption_train_path']

    mini_batch_loss, encoder, decoder = caption_train(vocab_path, image_path, config, caption_path,word2idx_path=word2idx_path)


    datestr = date2str()
    save_config(config, "config" + datestr, save_path)
    save_loss(mini_batch_loss,"loss"+datestr,save_path)
    save_model(encoder,"encoder"+datestr,save_path)
    save_model(decoder,"decoder"+datestr,save_path)

#print(date2str())
#train_test()
#train_procedure_test()

def yye_caption_train_test():
    config = get_config_yye()

    vocab_path = config['caption_vocab_path']
    word2idx_path = config['word2idx_train_path']
    image_path = config['caption_train_image_path']
    save_path = config['checkpoints_saved_path']
    caption_path = config['caption_train_path']

    #mini_batch_loss, encoder, decoder = attention_caption_train(vocab_path, image_path, config, caption_path,word2idx_path=word2idx_path)

    datestr = date2str()
    save_config(config, "attention_config" + datestr, save_path)
    #save_loss(mini_batch_loss, "attention_loss" + datestr, save_path)
    #save_model(encoder, "attention_encoder" + datestr, save_path)
    #save_model(decoder, "attention_decoder" + datestr, save_path)

# yye_caption_train_test()



def caption_test_test():
    config = get_config()
    vocab_path = config['caption_vocab_path']
    image_path = config['caption_train_image_path']
    encoder_path = config['caption_encoder_path']
    decoder_path = config['caption_decoder_path']
    caption_path = config['caption_test_path']
    config_path = config['config_path']
    word2idx_path = config['word2idx_test_path']
    batch = 1
    max_sequence_len = 30

    #test(vocab_path,encoder_path,decoder_path,caption_path,image_path,config_path,batch,max_sequence_len,word2idx_path=None)
    images,result_captions,original_captions = caption_test(vocab_path, encoder_path, decoder_path, caption_path, image_path, config_path, batch, max_sequence_len,
                                                            word2idx_path=word2idx_path)
    print(result_captions)
    visualize_img_caption(images, result_captions)

def kogpt_test():
    config = get_config()
    tok_path = get_tokenizer()
    model, vocab = get_pytorch_kogpt2_model()
    tok = SentencepieceTokenizer(tok_path)
    sent = '나는 밥을 먹었'
    toked = tok(sent)
    input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
    output = model(input_ids=input_ids)
    while 1:
        input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
        pred = model(input_ids)[0]
        gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
        print(gen)
        if gen == '</s>':
            break
        sent += gen.replace('▁', ' ')
        toked = tok(sent)
    print(sent)

def korean_gpt_long_setence_life_test():
    config = get_config()
    kogpt2_config = get_kog_config()
    kogpt2_model_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\checkpoints\\kogpt_life_model_20_2020-04-26-23-56-31.pth"

    kogpt2_vocab_path = config['kogpt_vocab_path']
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    torch.load(kogpt2_model_path)
    kogpt2model.load_state_dict(torch.load(kogpt2_model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kogpt2model.to(device)
    kogpt2model.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')
    tok = SentencepieceTokenizer(kogpt2_vocab_path)

    sent = '나는 밥을 먹었'
    toked = tok(sent)
    print(toked)
    sent_cnt = 0

    input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
    input_ids = input_ids.to(device)

    outputs = kogpt2model.generate(
        input_ids=input_ids,
        max_length=100, min_length=50,repetition_penalty=1.2, do_sample=True,num_beams=3,bos_token_id=0,pad_token_id=3,eos_token_id=1, num_return_sequences=3)

    target = outputs[0]
    print("========수필===========")
    for i in range(3):  # 3 output sequences were generated
        toked = vocab.to_tokens(outputs[i].squeeze().tolist())
        ret = re.sub(r'(<s>|</s>|<pad>|<unk>)', '', ''.join(toked).replace('▁', ' ').strip())
        print('Generated {}: {}'.format(i, ret))

def korean_gpt_long_setence_story_test():
    config = get_config()
    kogpt2_config = get_kog_config()
    kogpt2_model_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\checkpoints\\kogpt_story_model_30_2020-04-28-09-32-34.pth"
    kogpt2_vocab_path = config['kogpt_vocab_path']
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    kogpt2model.load_state_dict(torch.load(kogpt2_model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kogpt2model.to(device)
    kogpt2model.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')
    tok = SentencepieceTokenizer(kogpt2_vocab_path)

    sent = '나는 밥을 먹었'
    toked = tok(sent)
    print(toked)
    sent_cnt = 0

    input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
    input_ids = input_ids.to(device)

    outputs = kogpt2model.generate(
        input_ids=input_ids,
        max_length=100, min_length=50,repetition_penalty=1.2, do_sample=True,num_beams=3,bos_token_id=0,pad_token_id=3,eos_token_id=1, num_return_sequences=3)

    target = outputs[0]
    print("=========소설=========")

    for i in range(3):  # 3 output sequences were generated
        toked = vocab.to_tokens(outputs[i].squeeze().tolist())
        ret = re.sub(r'(<s>|</s>|<pad>|<unk>)', '', ''.join(toked).replace('▁', ' ').strip())
        print('Generated {}: {}'.format(i, ret))

def korean_gpt_short_setence_life_test():
    config = get_config()
    kogpt2_config = get_kog_config()
    kogpt2_model_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\checkpoints\\kogpt_life_model_20_2020-04-26-23-56-31.pth"

    kogpt2_vocab_path = config['kogpt_vocab_path']
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    torch.load(kogpt2_model_path)
    kogpt2model.load_state_dict(torch.load(kogpt2_model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kogpt2model.to(device)
    kogpt2model.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')
    tok = SentencepieceTokenizer(kogpt2_vocab_path)

    sent = '나는 밥을 먹었'
    toked = tok(sent)
    print(toked)
    sent_cnt = 0

    input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
    input_ids = input_ids.to(device)

    outputs = kogpt2model.generate(
        input_ids=input_ids,
        repetition_penalty=1.2, do_sample=True,bos_token_id=0,pad_token_id=3,eos_token_id=1, num_return_sequences=1)


    print("========수필===========")
    for i in range(1):  # 3 output sequences were generated
        toked = vocab.to_tokens(outputs[i].squeeze().tolist())
        ret = re.sub(r'(<s>|</s>|<pad>|<unk>)', '', ''.join(toked).replace('▁', ' ').strip())
        print('Generated {}: {}'.format(i, ret))

def korean_gpt_short_setence_story_test():
    config = get_config()
    kogpt2_config = get_kog_config()
    kogpt2_model_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\checkpoints\\kogpt_story_model_30_2020-04-28-09-32-34.pth"
    kogpt2_vocab_path = config['kogpt_vocab_path']
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    kogpt2model.load_state_dict(torch.load(kogpt2_model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kogpt2model.to(device)
    kogpt2model.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')
    tok = SentencepieceTokenizer(kogpt2_vocab_path)

    sent = '나는 밥을 먹었'
    toked = tok(sent)
    print(toked)
    sent_cnt = 0

    input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
    input_ids = input_ids.to(device)

    outputs = kogpt2model.generate(
        input_ids=input_ids,
        repetition_penalty=1.2, do_sample=True, bos_token_id=0, pad_token_id=3,
        eos_token_id=1, num_return_sequences=1)

    print("========소설===========")
    for i in range(1):  # 3 output sequences were generated
        toked = vocab.to_tokens(outputs[i].squeeze().tolist())
        ret = re.sub(r'(<s>|</s>|<pad>|<unk>)', '', ''.join(toked).replace('▁', ' ').strip())
        print('Generated {}: {}'.format(i, ret))

def preprocessed_test():
    path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\kogpt\\life_pkl.pkl"
    with open(path, "rb") as handle:
        examples = pickle.load(handle)
    print(examples)
    print(examples)

def fine_tune_test():
    AI_DIRECTORY = "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI"
    #AI_DIRECTORY = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI"
    config = get_config()
    fine_tune_num = 0
    new_kogpt_model,loss_record = fine_tuning(config,fine_tune_num,AI_DIRECTORY)
    epoch=config['kogpt_epoch']
    datestr = date2str()
    default_name = "kogpt_story_"
    model_name = default_name+"model_"+str(epoch)+"_"+datestr
    loss_name = default_name+"loss_"+datestr
    root_path = AI_DIRECTORY + config['checkpoints_saved_path']
    save_model(new_kogpt_model,model_name,root_path)
    save_loss(loss_record,loss_name,root_path)

def predction_test(images,model_type="life"):
    root_path = "C:\\Users\\multicampus\\Downloads\\images\\test"
    AI_directory_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI"
    ret = predict(images,root_path,AI_directory_path,model_type=model_type)
    return ret

def build_korean_to_idx_test():
    config = get_config()
    #AI_DIRECTORY = "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI"
    AI_DIRECTORY = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI"
    kogpt2_vocab_path = AI_DIRECTORY + config['kogpt_vocab_path']
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')
    tok = SentencepieceTokenizer(kogpt2_vocab_path)
    file_path =AI_DIRECTORY + "\\datasets\\kogpt\\story_train_pkl.pkl"
    save_path = AI_DIRECTORY + "\\datasets\\kogpt\\"
    build_korean_to_idx(file_path,save_path,vocab,tok,block_size=256)

def Kogpt2Dataset_TEST():
    file_path = "C:\\Users\\multicampus\\s02p23c104\\Back\AI\\datasets\\kogpt\\life_to_idx.pkl"
    kogpt2_dataset = Kogpt2Dataset(file_path)

def kogpt_life_recursive_test():
    config = get_config()
    AI_directory_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI"
    kogpt2_config = get_kog_config()
    kogpt2_model_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\checkpoints\\kogpt_life_model_20_2020-04-26-23-56-31.pth"

    kogpt2_vocab_path = AI_directory_path+config['kogpt_vocab_path']
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    torch.load(kogpt2_model_path)
    kogpt2model.load_state_dict(torch.load(kogpt2_model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kogpt2model.to(device)
    kogpt2model.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')
    tok = SentencepieceTokenizer(kogpt2_vocab_path)

    # sent = ' 신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있자, 신부님은 웨딩드레파란 셔츠를 입은 남자가 사다리에 서 있'
    # toked = tok(sent)
    # print(toked)
    # sent_cnt = 0

    # input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
    # input_ids = input_ids.to(device)
    korean_sentences = ['신랑 신부가 결혼식 파티 앞에서 사진을 찍기 위해 포즈를 취하고 있다.', '파란 셔츠를 입은 남자가 사다리에 서 있다.', '두 남자가 서 있다']
    kogpt_input_sentences = []
    for korean in korean_sentences:
        korean_size = len(korean)
        if not kogpt_input_sentences:
            korean_size = len(korean)
            if korean_size > 3:
                kogpt_input_sentences.append(korean[:-2])
            elif korean_size > 1:
                kogpt_input_sentences.append(korean[:-1])
            else:
                kogpt_input_sentences.append(korean)
        else:
            for i in range(len(kogpt_input_sentences)):
                if korean_size > 3:
                    kogpt_input_sentences[i] += korean[:-2]
                elif korean_size > 1:
                    kogpt_input_sentences[i] += korean[:-1]
                else:
                    kogpt_input_sentences[i] += korean[:]
        kogpt_output_sentences = []
        print(kogpt_input_sentences)
        expected_length = 50
        for kogpt_input_sentence in kogpt_input_sentences:
            print(kogpt_input_sentence)
            toked = tok(kogpt_input_sentence)
            input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
            print(input_ids)
            input_ids = input_ids.to(device)
            input_length = input_ids.shape[1]
            outputs = kogpt2model.generate(
                input_ids=input_ids,max_length=input_length+expected_length,repetition_penalty=1.2, do_sample=True, num_beams=3, bos_token_id=0,
                pad_token_id=3, eos_token_id=1, num_return_sequences=3)
            for i in range(3):  # 3 output sequences were generated
                toked = vocab.to_tokens(outputs[i].squeeze().tolist())
                ret = re.sub(r'(<s>|</s>|<pad>|<unk>|)', '', ''.join(toked).replace('▁', ' ').strip())
                kogpt_output_sentences.append(ret)
        kogpt_input_sentences = copy.deepcopy(kogpt_output_sentences)
    print(kogpt_input_sentences)

    # outputs = kogpt2model.generate(
    #     input_ids=input_ids,
    #     max_length=100, min_length=50, repetition_penalty=1.2, do_sample=True, num_beams=3, bos_token_id=0,
    #     pad_token_id=3, eos_token_id=1, num_return_sequences=3)
    #
    # target = outputs[0]
    # print("========수필===========")
    # for i in range(3):  # 3 output sequences were generated
    #     toked = vocab.to_tokens(outputs[i].squeeze().tolist())
    #     ret = re.sub(r'(<s>|</s>|<pad>|<unk>)', '', ''.join(toked).replace('▁', ' ').strip())
    #     print('Generated {}: {}'.format(i, ret))

def string_test():
    a = [['aaa.', 'bbb.', 'ccc.'], ['ddd.', 'eee', 'fff.']]
    korean_postprocess(a)
    print(a)

def attenntion_training_test():
    config = get_config()
    print(config)
    AI_DIREC = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI"
    vocab_path = AI_DIREC + config['caption_vocab_path']
    word2idx_path = AI_DIREC + config['word2idx_train_path']
    image_path = config['caption_train_image_path']
    save_path = AI_DIREC + config['checkpoints_saved_path']
    print(save_path)
    caption_path = AI_DIREC + config['caption_train_path']

    print(vocab_path)
    print(word2idx_path)
    mini_batch_loss, encoder, decoder = attention_caption_train(vocab_path, image_path, config, caption_path,
                                                      word2idx_path=word2idx_path)

    datestr = date2str()
    save_config(config, "attention_config" + datestr, save_path)
    save_loss(mini_batch_loss, "attention_loss" + datestr, save_path)
    save_model(encoder, "attention_encoder" + datestr, save_path)
    save_model(decoder, "attention_decoder" + datestr, save_path)

def attention_beam_search_test(images,root_path):
    config = get_config()
    # 0. Extract captions from images
    AI_directory_path = "C:\\Users\\multicampus\\s02p23c104\\Back\\AI"
    vocab = load_voca(AI_directory_path + config['caption_attention_vocab_path'])
    emb_dim = config['caption_embed_size']
    decoder_dim = config['caption_hidden_size']
    attention_dim = config['caption_attention_dim']
    dropout = config['caption_dropout_ratio']
    caption_encoder_path = AI_directory_path + config['caption_attention_encoder_path']
    caption_decoder_path = AI_directory_path + config['caption_attention_decoder_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_sequence_len = 50  # default value

    transform = torch_transform.Compose([
        torch_transform.ToTensor(),
        torch_transform.Normalize(mean=(0.4444, 0.4215, 0.3833), std=(0.2738, 0.2664, 0.2766))])

    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(vocab),
                                   dropout=dropout)

    encoder.load_state_dict(torch.load(caption_encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(caption_decoder_path, map_location=device))
    images = load_image(images, root_path, transform)

    encoder.eval()
    decoder.eval()

    encoder.to(device)
    decoder.to(device)
    images = images.to(device)
    batch = images.shape[0]

    predicted_index = []
    encoder_out = encoder(images)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(batch, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)
    k_prev_words=torch.LongTensor([[vocab('<start>')]] * batch).to(device)
    h, c = decoder.init_hidden_state(encoder_out)
    for i in range(max_sequence_len):
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        scores = decoder.fc(h)  # (s, vocab_size)
        _,predicted = scores.max(1)
        predicted_index.append(predicted)
        k_prev_words = predicted.unsqueeze(1)

    predicted_index = torch.stack(predicted_index, dim=1)
    predicted_index = predicted_index.cpu().numpy()

    result_captions = []
    for wordindices in predicted_index:
        text = ""
        for index in wordindices:
            word = vocab.idx2word[index]
            if word == '<end>':
                break
            if word == '<unk>' or word == '<start>':
                continue
            text += word + " "
        result_captions.append(text)

    print("result_caption : ", result_captions)



#test_test()
#loader_test()
#kogpt_test()
#preprocessed_test()
#our_gpt_model_test()
#caption_test_test()
#build_voca_test()
#caption_train_test()
#train_procedure_test()
#print(predction_test())
#fine-tune process
#build_korean_to_idx_test()
#Kogpt2Dataset_TEST()
#fine_tune_test()

#korean_gpt_long_setence_life_test()
#korean_gpt_long_setence_story_test()

#korean_gpt_short_setence_life_test()
#korean_gpt_short_setence_story_test()

# images =["65567.jpg","81641.jpg","574181.jpg"]
# print("======수필======")
# ret = predction_test(images,"life")
# for content in ret:
#     print(content)
# print("======소설======")
# ret = predction_test(images,"story")
# for content in ret:
#     print(content)
# print("======기사=====")
# ret = predction_test(images,"news")
# for content in ret:
#     print(content)
# print("=====================================================")
# images =["667626.jpg","675153.jpg","726414.jpg"]
# print("======수필======")
# ret = predction_test(images,"life")
# for content in ret:
#     print(content)
# print("======소설======")
# ret = predction_test(images,"story")
# for content in ret:
#     print(content)
# print("======기사=====")
# ret = predction_test(images,"news")
# for content in ret:
#     print(content)
# print("=====================================================")
# images =["764507.jpg","793558.jpg","807129.jpg"]
# print("======수필======")
# ret = predction_test(images,"life")
# for content in ret:
#     print(content)
# print("======소설======")
# ret = predction_test(images,"story")
# for content in ret:
#     print(content)
# print("======기사=====")
# ret = predction_test(images,"news")
# for content in ret:
#     print(content)
#kogpt_life_recursive_test()


#fine_tune_test()
#save_caption2idx_test()
images =["667626.jpg","675153.jpg","726414.jpg"]
root_path = "C:\\Users\\multicampus\\Downloads\\images\\test"
attention_beam_search_test(images,root_path)


#attenntion_training_test()