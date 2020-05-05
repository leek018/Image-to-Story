#  Result

![65567](https://user-images.githubusercontent.com/37829888/81047651-d59a1c80-8ef5-11ea-8a45-29fb49145960.jpg)

신랑 신부가 춤을 추고 있자, 나는 신부 쪽을 흘깃 보았다. 그 애는 허리를 구부린 채 양 손을 머리 위로 끌어 올렸다가 다시 뒤로 뺐다 반복했다. 그러더니 이내 자세를 고쳐 앉았다. 그러자 신부 쪽에서도 춤추는 것 같은 포즈가 나타났다. 신부 쪽이 더 자세하게 몸짓을 흉내내려는 찰나에, 그 애의 입술이 턱에서 미끄러져 나왔다. 순간적으로 당황한 모양이었다. 신부 쪽도 어쩔 수 없다는 표정이었다.

#  Architecture

![ssafy_architecture](https://user-images.githubusercontent.com/37829888/81047809-2742a700-8ef6-11ea-96bc-e44a7c9df6d2.PNG)



# REQURIED LIBRARY

+ cuda >= 10.1
+ pytorch version  >=1.4 확인
  + pip show torch
  + 만약 1.4이상이 아니라면 conda uninstall pytorch
+ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch or pip install pytorch torchvision
+ pip 설치할 때는 꼭 가상환경에서 실행시켜야 합니다!!
  + pip install nltk
  + pip install pandas
  + pip install setuptools
  + pip install mxnet==1.6.0
  + pip install gluonnlp ==0.8.3
  + pip install sentencepiece
  + pip install transformers

# install

+ conda 환경이라면 

  + conda actviate pytorch_env
+ pip install .
  + 단 python 버전이 3.6 이상이여야 합니다 꼭!



# Run

prediction.py

+ predict(images,IMAGE_DIRECTORY,AI_DIRECTORY_PATH,model_type)
  + param 
    + images : 이미지 배열
    + IMAGE_DIRECTORY: images가 저장된 디렉토리
    + AI_DIRECTORY_PATH: AI DIRECTORY path(절대경로)
    + model_type(string)
      + "life","story","news"
  + return
    + 1가지 이상의 생성된 문장 배열