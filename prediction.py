import torch
from AI.config import get_config,get_kog_config
from AI.models.Captionmodel.Encoder import EncoderCNN
from AI.models.Captionmodel.Decoder import DecoderRNN
from AI.vocab.CaptionVocab import load_voca
from PIL import Image
from torchvision import transforms as torch_transform
from AI.api.papago_api import get_translate
from transformers import GPT2Config, GPT2LMHeadModel
import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
import re
import copy
def load_image(images,root_path, transform=None):
    buffer=[]
    for image_name in images:
        path = root_path+"/"+image_name
        image = Image.open(path).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        if transform is not None:
            image = transform(image).unsqueeze(0)
            buffer.append(image)
    torch_images = torch.cat(buffer,dim=0)
    return torch_images

def recursive_prediction(korean_sentences,tok,vocab,device,kogpt2model):
    kogpt_input_sentences = []
    for korean in korean_sentences:
        korean_size = len(korean)
        if not kogpt_input_sentences:
            korean_size = len(korean)
            if korean_size > 3:
                kogpt_input_sentences.append(korean[:-1])
            elif korean_size > 1:
                kogpt_input_sentences.append(korean[:-1])
            else:
                kogpt_input_sentences.append(korean)
        else:
            for i in range(len(kogpt_input_sentences)):
                if korean_size > 3:
                    kogpt_input_sentences[i] += korean[:-1]
                elif korean_size > 1:
                    kogpt_input_sentences[i] += korean[:-1]
                else:
                    kogpt_input_sentences[i] += korean[:]
        kogpt_output_sentences = []
        print(kogpt_input_sentences)
        for kogpt_input_sentence in kogpt_input_sentences:
            toked = tok(kogpt_input_sentence)
            input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
            input_ids = input_ids.to(device)
            # outputs = kogpt2model.generate(
            #     input_ids=input_ids,max_length=80, repetition_penalty=1.2, do_sample=True, num_beams=3, bos_token_id=0, pad_token_id=3,eos_token_id=1, num_return_sequences=3)
            outputs = kogpt2model.generate(
                input_ids=input_ids, repetition_penalty=1.2, do_sample=True, num_beams=3, bos_token_id=0,
                pad_token_id=3, eos_token_id=1, num_return_sequences=3)
            for i in range(3):  # 3 output sequences were generated
                toked = vocab.to_tokens(outputs[i].squeeze().tolist())
                ret = re.sub(r'(<s>|</s>|<pad>|<unk>|)', '', ''.join(toked).replace('▁', ' ').strip())
                kogpt_output_sentences.append(ret)
        kogpt_input_sentences = copy.deepcopy(kogpt_output_sentences)

def korean_preprocess(korean_sentences):
    for i in range(len(korean_sentences)):
        korean_size = len(korean_sentences[i])
        if korean_size > 1:
            korean_sentences[i] = korean_sentences[i][:-1]
def make_sentence(inputs,sentence,result,start):
    if start == len(inputs):
        result.append(sentence)
        return
    for sent in inputs[start]:
        make_sentence(inputs,sentence+sent,result,start+1)
    return

def korean_postprocess(gpt_result):
    for i in range(len(gpt_result)):
        for j in range(len(gpt_result[i])):
            find_point_result = gpt_result[i][j].rfind('.')
            if find_point_result != -1:
                gpt_result[i][j] = gpt_result[i][j][1:find_point_result+1] + '\n\n'
            else:
                gpt_result[i][j] += '\n\n'

def naive_prediction(korean_sentences,tok,vocab,device,kogpt2model,model_type):
    result =[]
    for sent in korean_sentences:
        temp = []
        toked = tok(sent)
        input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
        input_ids = input_ids.to(device)
        if model_type == "news":
            outputs = kogpt2model.generate(
                input_ids=input_ids, repetition_penalty=1.2, do_sample=True, num_beams=3, bos_token_id=0,
                pad_token_id=3, eos_token_id=1, num_return_sequences=3)
        else:
            outputs = kogpt2model.generate(
                input_ids=input_ids,
                max_length=100, min_length=50, repetition_penalty=1.2, do_sample=True, num_beams=3, bos_token_id=0,
                pad_token_id=3, eos_token_id=1, num_return_sequences=3)

        for i in range(3):  # 3 output sequences were generated
            toked = vocab.to_tokens(outputs[i].squeeze().tolist())
            ret = re.sub(r'(<s>|</s>|<pad>|<unk>)', '', ''.join(toked).replace('▁', ' ').strip())
            temp.append(ret)
        result.append(temp)
    return result

def predict(images,root_path,AI_directory_path,model_type="life"):
    config = get_config()
    #0. Extract captions from images
    vocab = load_voca(AI_directory_path+config['caption_vocab_path'])
    caption_embed_size = config['caption_embed_size']
    caption_hidden_layer = config['caption_hidden_layer']
    caption_hidden_size = config['caption_hidden_size']
    caption_encoder_path = AI_directory_path+config['caption_encoder_path']
    caption_decoder_path = AI_directory_path+config['caption_decoder_path']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_sequence_len = 30 #default value

    transform = torch_transform.Compose([
        torch_transform.ToTensor(),
        torch_transform.Normalize(mean=(0.4444, 0.4215, 0.3833), std=(0.2738, 0.2664, 0.2766))])

    encoder = EncoderCNN(caption_embed_size)
    decoder = DecoderRNN(caption_embed_size, len(vocab), caption_hidden_layer, caption_hidden_size)

    encoder.load_state_dict(torch.load(caption_encoder_path,map_location=device))
    decoder.load_state_dict(torch.load(caption_decoder_path,map_location=device))
    images = load_image(images, root_path, transform)

    encoder.eval()
    decoder.eval()

    encoder.to(device)
    decoder.to(device)
    images = images.to(device)

    features = encoder(images)
    states = None
    predicted_index = []
    lstm_inputs = features.unsqueeze(1)

    for i in range(max_sequence_len):
        outputs,states = decoder.lstm(lstm_inputs,states)
        # outputs을 linear 레이어의 인풋을 위해 2차원 배열로 만들어 줘야함
        outputs = outputs.squeeze(1)
        scores_per_batch = decoder.score_layer(outputs)
        values, predicted = scores_per_batch.max(1)
        predicted_index.append(predicted)
        lstm_inputs = decoder.embed(predicted)
        lstm_inputs = lstm_inputs.unsqueeze(1)

    predicted_index = torch.stack(predicted_index,dim=1)
    predicted_index = predicted_index.cpu().numpy()

    result_captions = []
    for wordindices in predicted_index:
        text =""
        for index in wordindices:
            word = vocab.idx2word[index]
            if word == '<end>':
                break
            if word == '<unk>' or word == '<start>':
                continue
            text += word + " "
        result_captions.append(text)

    print("result_caption : ",result_captions)
    # 1. translate captions to korean

    korean_sentences = []
    for sent in result_captions:
        translate_result = get_translate(sent)
        if translate_result != -1:
            translate_result = re.sub(r'\.','',translate_result)
            korean_sentences.append(translate_result)
    print("result_korean : ",korean_sentences)

    kogpt2_config = get_kog_config()
    if model_type == "life":
        kogpt2_model_path = AI_directory_path+config['kogpt_life_model_path']
    elif model_type == "story":
        kogpt2_model_path = AI_directory_path + config['kogpt_story_model_path']
    else:
        kogpt2_model_path = AI_directory_path+config['kogpt_model_path']
    kogpt2_vocab_path = AI_directory_path+config['kogpt_vocab_path']
    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    kogpt2model.load_state_dict(torch.load(kogpt2_model_path,map_location=device))

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

    korean_preprocess(korean_sentences)
    gpt_result = naive_prediction(korean_sentences,tok,vocab,device,kogpt2model,model_type)
    korean_postprocess(gpt_result)
    result = []
    make_sentence(gpt_result,"",result,0)
    result.sort(key=lambda item: (-len(item),item))
    result_len = len(result)
    if result_len >11:
        result_len = 11
    result = result[1:result_len]
    return result

