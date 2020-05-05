from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import AI.config as cfg
import torch
from AI.preprocess.Caption_preprocess import *
import pickle
import datetime


def save_model(model,name,root_path):
	path = root_path+name+".pth"
	torch.save(model.state_dict(),path)

def save_config(config,name,root_path):
	save_path = root_path+name+".json"
	with open(save_path, 'w', encoding='utf-8') as f:
		json.dump(config, f)


def save_loss(loss,name,root_path):
	save_path = root_path+name+".pkl"
	with open(save_path, 'wb') as f:
		pickle.dump(loss, f)

def visualize_img_caption(images,captions):
	fig = plt.figure()
	length = len(images)
	for i in range(length):
		subplot_img = fig.add_subplot(length, 2, i+1)
		npimg = images[i].numpy()
		npimg = npimg.squeeze()
		subplot_img.imshow(np.transpose(npimg,(1,2,0)))

		subplot_text = fig.add_subplot(length, 2, i + 2)
		subplot_text.text(0.5,0.5,captions[i])
	plt.show()

def date2str():
	return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
