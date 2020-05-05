import torch
import torch.nn as nn
import torchvision.models as models

# transfer learning 전략을 사용합니다.
# transfer learning 이란 이미 학습된 네트워크 혹은 모델을 가지고
# 새로운 네트워크를 만드는데 집중하는 것입니다.
# 이러한 전략을 통해 시간을 줄일 수 있습니다.
# 여기서는 ImageNet으로 학습된 resnet-152를 사용할 것이고
# classification을 수행하지 않을 것이기 때문에
# 맨마지막 fullyconnected layer를 제외 시킬 것입니다.
# resnet-152의 마지막 레이어는 참고로 fc-1000(1000개짜리 perceptron)을 가집니다.
# 이름 제거하고 fc-embed_size를 추가합니다.
# 여기서 embed_size 란 다음 디코더의 입력으로 들어갈 vector의 길이 입니다.
class EncoderCNN(nn.Module):

    def __init__(self,embed_size):
        super(EncoderCNN,self).__init__()
        resnet = models.resnet152(pretrained=True)
        network_list = list(resnet.children())[:-1]
        self.network = nn.Sequential(*network_list) # python asterisk : unpacking list 할 때도 쓰인다.
        self.linear = nn.Linear(resnet.fc.in_features,embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self,x):
        with torch.no_grad():
            out = self.network(x)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        out = self.bn(out)
        return out