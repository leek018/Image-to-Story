import logging
import torch
from transformers import AdamW,get_linear_schedule_with_warmup,GPT2Config, GPT2LMHeadModel
from AI.config import get_kog_config
#KOGPT
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp as nlp

#fine tune
from AI.dataload.Kogpt2DataManagement import make_kogpt2_loader
from tqdm import tqdm, trange
import json
def freeze_model(pos,model):
    for p in model.parameters():
        p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
    length = len(list(model.children()))
    print(model.childer)
    for c in list(model.children())[length-pos:length]:
        for p in c.parameters():
            p.requires_grad = True

def fine_tuning(config,fine_tune_num,AI_DIRECTORY):
    """ Train the model """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = config['kogpt_batch_size']
    train_path = AI_DIRECTORY + config['kogpt_story_train_data_path']
    num_train_epochs = config['kogpt_epoch']

    kogpt2_config = get_kog_config()
    kogpt2_model_path = AI_DIRECTORY + config['kogpt_model_path']
    kogpt2_vocab_path = AI_DIRECTORY + config['kogpt_vocab_path']

    kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
    kogpt2model.load_state_dict(torch.load(kogpt2_model_path))

    kogpt2model.to(device)

    vocab = nlp.vocab.BERTVocab.from_sentencepiece(kogpt2_vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')
    tok = SentencepieceTokenizer(kogpt2_vocab_path)

    loader = make_kogpt2_loader(train_path,batch_size)
    num_training_steps = len(loader)*num_train_epochs
    learning_rate = 5e-6
    adam_epsilon = 1e-8
    warmup_steps = 0
    no_decay = ["bias", "LayerNorm.weight"]
    #freeze_model(fine_tune_num,kogpt2model)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in kogpt2model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in kogpt2model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    global_step = 0
    epochs_trained = 0

    tr_loss = 0.0
    logging_loss = 0.0
    kogpt2model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch"
    )
    logging_steps  = 500
    loss_record = []
    for epoch in train_iterator:
        epoch_iterator = tqdm(loader, desc="Iteration")
        for step, inputs in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training

            ###############
            kogpt2model.train()
            input = inputs.to(device)
            label = inputs.to(device)

            outputs = kogpt2model(input_ids=input,labels=label)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            kogpt2model.zero_grad()
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

    return kogpt2model,loss_record