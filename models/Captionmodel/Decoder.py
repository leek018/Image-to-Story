import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class DecoderRNN(nn.Module):
    def __init__(self,embed_size,vocab_size,hidden_layers_num, hidden_size):
        super(DecoderRNN,self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_layers_num = hidden_layers_num
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_size)
        # input의 차원중 제일 앞의 차원이 batch_size가 오는 것을 알립니다.
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=hidden_layers_num,batch_first=True)
        # 인풋으로 넣은 단어가 전체 vocab에서 어떤 단어에 해당하는지 판단하는데 사용될 layer
        self.score_layer = nn.Linear(hidden_size,vocab_size)

    def forward(self,word2idx,feature,length):
        # feature는 image로 부터 얻은 feature값 shape[ batch X embed_size]
        # ex) [ batch_size X embed_size] => [10 * 10]

        # word2idx의 값들을 embeding layer의 lookup table을 참고하여 변환합니다.
        idx2embeddings = self.embed(word2idx)

        # feature와 임베딩된 결과를 합쳐 줍니다.
        # 합쳐주는 과정은 sub2_test package의 embed_feature_cat_test() 에서 확인 할 수 있습니다.
        idx2embeddings = torch.cat([feature.unsqueeze(1),idx2embeddings],1)

        # optional
        # 효율적인 실행을 위해 pack_padded sequence를 사용합니다.
        # pack_padded sequence은 legth를 참고하여 매번 모든 batch들을 실행하지 않도록 합니다.
        # 예를 들어, 문장들이
        # [1,2,3,4,0]
        # [1,2,4,0,0]
        # [1,4,0,0,0]
        # 이런 word2idx가 있다고 합시다.
        # 4가 < end > 를 의미하고 , 0 가 <pad>를 의미합니다.
        # 첫 번째 word2idx는 3번째 요소인 3의 값 까지 만을 입력으로 넣어주면 됩니다.
        # 두 번째 word2idx는 2번째 요소인 2의 값 까지 만을 입력으로 넣어주면 됩니다.
        # 세 번째 word2idx는 1번째 요소인 1의 값 까지 만을 입력으로 넣어주면 됩니다.

        pack = pack_padded_sequence(idx2embeddings,length,batch_first=True)

        # pack
        # pack을 하지 않았다면 output = self.lstm(embedding)이 됩니다.
        output,(ht,ct) = self.lstm(pack) #lstm의 결과는 출력, hidden_state, cell_state가 나옵니다.
        output = self.score_layer(output.data)

        #output shpae : [문장의 모든 길이 X vocab_size] => 한 단어를 넣었을 때 vocab중 어떤 단어가 나올지에 대한 score
        return output