from AI.vocab.CaptionVocab import load_voca,load_tokenized_data,save_tokenized_data,tokenized_data
from AI.dataload.CaptionDataManagement import make_caption_loader
from AI.models.Captionmodel.models import Encoder
from AI.models.Captionmodel.models import DecoderWithAttention
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import json

def attention_caption_train(vocab_path, image_path, cfg, caption_path, word2idx_path=None):
    voca = load_voca(vocab_path)
    if word2idx_path is not None:
        dataset = load_tokenized_data(word2idx_path)
    else:
        dataset = tokenized_data(caption_path, voca, type="train")
        save_tokenized_data(dataset,type="train")

    batch = cfg['caption_batch']
    emb_dim = cfg['caption_embed_size']
    decoder_dim = cfg['caption_hidden_size']
    attention_dim = cfg['caption_attention_dim']
    dropout = cfg['caption_dropout_ratio']
    epochs = cfg['caption_epoch']
    loader = make_caption_loader(dataset, batch, image_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()
    encoder.fine_tune(False)
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(voca),
                                   dropout=dropout)

    encoder.to(device)
    decoder.to(device)
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    loss_function = nn.CrossEntropyLoss()
    param_list = list(decoder.parameters())
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
        encoder.train()
        decoder.train ()
        for idx_of_batch,(images, word2idxes,length) in enumerate(epoch_iterator):
            length = torch.LongTensor(length).to(device)
            images,word2idxes = images.to(device),word2idxes.to(device)
            features = encoder(images)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, word2idxes, length)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = loss_function(scores, targets)

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