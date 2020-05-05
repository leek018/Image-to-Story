import json
import os
# Req. 2-1	Config.py 파일 생성
# config = {
#     #config path
#     'config_path' : "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\checkpoints\\config2020-04-04-04-48-01.json",
#     #checkpoints_saved_path
#     'checkpoints_saved_path':"\\checkpoints\\",
#     #caption_config
#     'caption_train_image_path' : "C:\\Users\\multicampus\\Downloads\\images\\train\\",
#     'caption_train_word_saved_path' : "\\datasets\\caption\\",
#     #'caption_vocab_path' : "\\datasets\\caption\\voca_30.pickle",
#     'caption_vocab_path' : "\\datasets\\caption\\voca_5.pickle",
#
#     'caption_batch' :  64,
#     'caption_embed_size':512,
#     'caption_hidden_layer':2,
#     'caption_hidden_size':512,
#     'caption_epoch':30,
#     'caption_optimizer' : 'SGD',
#
#
#     'caption_encoder_path' :"\\checkpoints\\encoder2020-04-28-13-36-26.pth",
#     'caption_decoder_path' :"\\checkpoints\\decoder2020-04-28-13-36-26.pth",
#
#
#     'caption_test_path' : "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\caption\\test_data.json",
#     'word2idx_test_path' : "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\datasets\\caption\\wordVector_test.pickle",
#     'caption_train_path' : "\\datasets\\caption\\train_data.json",
#     'word2idx_train_path' : "\\datasets\\caption\\wordVector_train_5.pickle",
#     'caption_attention_dim' : 512,
#     'caption_dropout_ratio':0.5,
#     #kogpt_config
#     'kogpt_batch_size':2,
#     'kogpt_life_train_data_path':"\\datasets\\kogpt\\life_to_idx.pkl",
#     'kogpt_story_train_data_path':"\\datasets\\kogpt\\story_to_idx.pkl",
#     'kogpt_epoch':30,
#     'kogpt_model_path' :"\\checkpoints\\pytorch_kogpt2_676e9bcfa7.params",
#     'kogpt_life_model_path' :"\\checkpoints\\kogpt_life_model_20_2020-04-26-23-56-31.pth",
#     'kogpt_story_model_path' :"\\checkpoints\\kogpt_story_model_30_2020-04-28-09-32-34.pth",
#     'kogpt_vocab_path' :"\\datasets\\kogpt\\kogpt2_news_wiki_ko_cased_818bfa919d.spiece",
#
#     # attention_config
#     'caption_attention_dim': 512,
#     'caption_attention_vocab_path': "/datasets/caption/voca_5.pickle",
#     'caption_attention_encoder_path': "/checkpoints/attention_encoder2020-05-01-02-28-24.pth",
#     'caption_attention_decoder_path': "/checkpoints/attention_decoder2020-05-01-02-28-24.pth",
#     'caption_dropout_ratio': 0.5,
# }

config = {
    #config path
    'config_path' : "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\checkpoints\\config2020-04-04-04-48-01.json",
    #checkpoints_saved_path
    'checkpoints_saved_path':"C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\checkpoints\\",
    #caption_config
    'caption_train_image_path' : "C:\\Users\\multicampus\\Downloads\\images\\train\\",
    'caption_train_word_saved_path' : "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\caption\\",
    'caption_vocab_path' : "/datasets/caption/voca_30.pickle",


    'caption_batch' :  32,
    'caption_embed_size':512,
    'caption_hidden_layer':2,
    'caption_hidden_size':512,
    'caption_epoch':20,
    'caption_optimizer' : 'SGD',


    'caption_encoder_path' :"/checkpoints/encoder2020-04-28-13-36-26.pth",
    'caption_decoder_path' :"/checkpoints/decoder2020-04-28-13-36-26.pth",


    'caption_test_path' : "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\caption\\test_data.json",
    'word2idx_test_path' : "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\datasets\\caption\\wordVector_test.pickle",
    'caption_train_path' : "C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\caption\\train_data.json",
    'word2idx_train_path' : "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\datasets\\caption\\wordVector_train.pickle",
    #kogpt_config
    'kogpt_batch_size':2,
    'kogpt_life_train_data_path':"C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\kogpt\\life_to_idx.pkl",
    'kogpt_story_train_data_path':"C:\\Users\\multicampus\\s02p23c104\\Back\\AI\\datasets\\kogpt\\story_to_idx.pkl",
    'kogpt_epoch':30,
    'kogpt_model_path' :"/checkpoints/pytorch_kogpt2_676e9bcfa7.params",
    'kogpt_life_model_path' :"/checkpoints/kogpt_life_model_20_2020-04-26-23-56-31.pth",
    'kogpt_story_model_path' :"/checkpoints/kogpt_story_model_30_2020-04-30-07-54-28.pth",
    'kogpt_vocab_path' :"/datasets/kogpt/kogpt2_news_wiki_ko_cased_818bfa919d.spiece",

    # attention_config
    'caption_attention_dim': 512,
    'caption_attention_vocab_path': "/datasets/caption/voca_5.pickle",
    'caption_attention_encoder_path': "/checkpoints/attention_encoder2020-05-01-02-28-24.pth",
    'caption_attention_decoder_path': "/checkpoints/attention_decoder2020-05-01-02-28-24.pth",
    'caption_dropout_ratio': 0.5,
}
yye_config = {
    'caption_vocab_path' : "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\datasets\\caption\\voca_30.pickle",
    'word2idx_train_path': "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\datasets\\caption\\wordVector_train.pickle",
    'caption_train_image_path' : "C:\\Users\\multicampus\\images\\train\\",
    'checkpoints_saved_path' : "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\checkpoints\\",
    'caption_train_path' : "C:\\Users\\multicampus\\yye\\s02p23c104\\Back\\AI\\datasets\\caption\\train_data.json",
    'caption_batch' :  32,
    'caption_embed_size':100,
    'caption_hidden_size':100,
    'caption_epoch':20,
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}
def get_config(cfg=None):
    global config
    if cfg is None:        
        return config
    with open(cfg,'r') as json_file:
        json_config = json.load(json_file)        
        return json_config

def get_config_yye(cfg=None):
    global yye_config
    if cfg is None:
        print(yye_config)
        return yye_config
    with open(cfg,'r') as json_file:
        json_config = json.load(json_file)
        print(json_config)
        return json_config

def get_kog_config():
    return kogpt2_config
