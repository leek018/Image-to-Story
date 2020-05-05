import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # 인코딩 된 이미지를 변환
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # 디코더의 출력을 변환
        self.full_att = nn.Linear(attention_dim, 1)  # softmax 될 값을 계산
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 가중치를 계산

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha