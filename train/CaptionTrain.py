from AI.vocab.CaptionVocab import load_voca,load_tokenized_data,save_tokenized_data,tokenized_data
from AI.dataload.CaptionDataManagement import make_caption_loader
from AI.models.Captionmodel.Encoder import EncoderCNN
from AI.models.Captionmodel.Decoder import DecoderRNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import json
def caption_train(vocab_path, image_path, cfg, caption_path, word2idx_path=None):
    voca = load_voca(vocab_path)
    if word2idx_path is not None:
        dataset = load_tokenized_data(word2idx_path)
    else:
        dataset = tokenized_data(caption_path, voca, type="train")
        save_tokenized_data(dataset, type="train")

    batch = cfg['caption_batch']
    embed_size = cfg['caption_embed_size']
    hidden_size = cfg['caption_hidden_size']
    hidden_layer = cfg['caption_hidden_layer']
    epochs = cfg['caption_epoch']
    loader = make_caption_loader(dataset, batch, image_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size,len(voca),hidden_layers_num=hidden_layer,hidden_size=hidden_size)

    encoder.to(device)
    decoder.to(device)
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    loss_function = nn.CrossEntropyLoss()
    param_list = list(encoder.linear.parameters()) + list(encoder.bn.parameters()) + list(decoder.parameters())
    optimizer = AdamW(param_list, lr=learning_rate, eps=adam_epsilon)
    num_training_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    global_step = 0
    epochs_trained = 0

    tr_loss = 0.0
    logging_loss = 0.0
    train_iterator = trange(
        epochs_trained, int(epochs), desc="Epoch"
    )
    logging_steps = 500
    loss_record = []
    for epoch in train_iterator:
        epoch_iterator = tqdm(loader, desc="Iteration")
        for idx_of_batch,(images, word2idxes,length) in enumerate(epoch_iterator):
            images,word2idxes = images.to(device),word2idxes.to(device)
            features = encoder(images)
            compare_targets = pack_padded_sequence(word2idxes,length,batch_first=True).data

            output = decoder(word2idxes,features,length)
            loss = loss_function(output,compare_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            global_step += 1
            if logging_steps > 0 and global_step % logging_steps == 0:
                logs = {}
                loss_scalar = (tr_loss - logging_loss) / logging_steps
                learning_rate_scalar = scheduler.get_last_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                loss_record.append(loss_scalar)
                logging_loss = tr_loss
                epoch_iterator.write(json.dumps({**logs, **{"step": global_step}}))
    return loss_record,encoder,decoder