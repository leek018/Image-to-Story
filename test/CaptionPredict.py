from AI.config import get_config
from AI.dataload.CaptionDataManagement import make_caption_loader
from AI.vocab.CaptionVocab import load_voca,load_tokenized_data,tokenized_data,save_tokenized_data
from AI.models.Captionmodel.Encoder import EncoderCNN
from AI.models.Captionmodel.Decoder import DecoderRNN
from AI.utils import visualize_img_caption
import torch

def caption_test(vocab_path, encoder_path, decoder_path, caption_path, image_path, config_path, batch, max_sequence_len, word2idx_path=None):
    vocab = load_voca(vocab_path)
    cfg = get_config(config_path)

    embed_size = cfg['caption_embed_size']
    vocab_size = len(vocab)
    hidden_layers_num = cfg['caption_hidden_layer']
    hidden_size = cfg['caption_hidden_size']

    if word2idx_path is not None:
        dataset = load_tokenized_data(word2idx_path)
    else:
        dataset = tokenized_data(caption_path, vocab, type="test")
        save_tokenized_data(dataset, type="test")


    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size,vocab_size,hidden_layers_num,hidden_size)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    encoder.eval()
    decoder.eval()


    loader = make_caption_loader(dataset,batch,image_path)



    test_data_iter = iter(loader)
    images,captions,length = test_data_iter.next()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_images = images.to(device)
    features = encoder(images)
    states = None

    # features의 형태는 (batch,embed_size)인 2차원입니다. 그러나
    # 이후 사용될 lstm은 input으로 (batch,num of embeddings,embed_size) 3차원 형태를 요구하기 때문에
    # features의 차원을 강제로 늘려줍니다.
    lstm_inputs = features.unsqueeze(1)
    predicted_index = []
    for i in range(max_sequence_len):
        outputs,states = decoder.lstm(lstm_inputs,states)
        # outputs을 linear 레이어의 인풋을 위해 2차원 배열로 만들어 줘야함
        outputs = outputs.squeeze(1)
        scores_per_batch = decoder.score_layer(outputs)
        values, predicted = scores_per_batch.max(1)
        predicted_index.append(predicted)
        lstm_inputs = decoder.embed(predicted)
        lstm_inputs = lstm_inputs.unsqueeze(1)

    # tensor를 포함한 그냥 1차원 짜리 리스트 [batch * max_sequence_len] => 2차원의 매트릭스 [batch X max_sequence_len] 바꿔줘야 함
    # ex)
    # predicted_index = [tensor([0,3,6]),tensor([1,4,7]),tensor([2,5,8])]
    # 이걸
    # [0,1,2]
    # [3,4,5]
    # [6,7,8] 이렇게 바꿔줘야 함
    # 2차원 짜리를 만들건데 기존의 리스트는 dim 0 방향이 되고(세로방향)
    # 새로 붙이는 리스트는 dim 1 방향으로 붙여야 함(가로 방향)

    predicted_index = torch.stack(predicted_index,dim=1)
    # 현재 tensor가 gpu에 있으므로 cpu로 옮겨서 연산을 해야함.
    predicted_index = predicted_index.cpu().numpy()

    result_captions = []
    for wordindices in predicted_index:
        caption = []
        for index in wordindices:
            word = vocab.idx2word[index]
            if word == '<end>':
                break
            if word == '<unk>' or word == '<start>':
                continue
            caption.append(word)
        result_captions.append(caption)

    return images,result_captions,captions



