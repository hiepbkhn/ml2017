import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    BertTokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

##
PRETRAINED_MODEL_NAME = "uer/t5-base-chinese-cluecorpussmall"
MODEL_DIR = "/kaggle/working/model"

####
USE_GPU = torch.cuda.is_available()

args_dict = dict(
    data_dir="/kaggle/working/data",  
    # model_name_or_path=PRETRAINED_MODEL_NAME,
    # tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

    # learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    # max_input_length=512,
    # max_target_length=4,
    # train_batch_size=8,
    # eval_batch_size=8,
    # num_train_epochs=4,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    # seed=42,
)

model_parser = argparse.ArgumentParser()
model_parser.add_argument("--model_name_or_path", default=PRETRAINED_MODEL_NAME, type=str, required=True, help="Path to pretrained model", )
model_parser.add_argument("--learning_rate", default=3e-4, type=float, required=False, help="learning_rate", )
model_parser.add_argument("--seed", default=42, type=int, required=False, help="seed", )
model_parser.add_argument("--max_input_length", default=512, type=int, required=False, help="max_input_length", )
model_parser.add_argument("--max_target_length", default=30, type=int, required=False, help="max_target_length", )
model_parser.add_argument("--train_batch_size", default=8, type=int, required=False, help="train_batch_size", )
model_parser.add_argument("--eval_batch_size", default=8, type=int, required=False, help="eval_batch_size", )
model_parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="num_train_epochs", )

model_args = model_parser.parse_args()

args_dict.update({
    "model_name_or_path":model_args.model_name_or_path ,
    "tokenizer_name_or_path": model_args.model_name_or_path ,
    "learning_rate":     model_args.learning_rate ,
    "seed":                 model_args.seed ,
    "max_input_length":  model_args.max_input_length,  
    "max_target_length": model_args.max_target_length, 
    "train_batch_size":  model_args.train_batch_size,
    "eval_batch_size":   model_args.eval_batch_size,
    "num_train_epochs":  model_args.num_train_epochs, # 3,5
    })
args = argparse.Namespace(**args_dict)
print(args)

set_seed(args.seed)

####
class TsvDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512):
        self.file_path = os.path.join(data_dir, type_path)
        
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, text, answer):
        input = f"{text}"
        target = f"{answer}"
        return input, target
  
    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                assert len(line) == 2
                assert len(line[0]) > 0
                assert len(line[1]) > 0

                text = line[0]
                answer = line[1]

                input, target = self._make_record(text, answer)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
####
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

        self.tokenizer = BertTokenizer.from_pretrained(hparams.tokenizer_name_or_path, is_fast=True)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked), 
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     self.log("val_loss", avg_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     self.log("test_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, type_path, args):
        return TsvDataset(
            tokenizer=tokenizer, 
            data_dir=args.data_dir, 
            type_path=type_path, 
            input_max_len=args.max_input_length,
            target_max_len=args.max_target_length)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                             type_path="train.tsv", args=self.hparams)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer, 
                                           type_path="dev.tsv", args=self.hparams)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.hparams.train_batch_size, 
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.hparams.eval_batch_size, 
                          num_workers=4)
    
#### TRAIN
# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     "/content/checkpoints", 
#     monitor="val_loss", mode="min", save_top_k=1
# )

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    # checkpoint_callback=checkpoint_callback,
)

model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

# model.tokenizer.save_pretrained(MODEL_DIR)
# model.model.save_pretrained(MODEL_DIR)

#### TEST
import textwrap
from tqdm.auto import tqdm
# from sklearn import metrics

tokenizer = model.tokenizer
test_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv", 
                          input_max_len=args.max_input_length, 
                          target_max_len=args.max_target_length) # 

test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4)

trained_model = model.model
if USE_GPU:
    trained_model.cuda()
    
trained_model.eval()

outputs = []
confidences = []
targets = []

for batch in tqdm(test_loader):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    outs = trained_model.generate(input_ids=input_ids, 
        attention_mask=input_mask, 
        max_length=args.max_target_length, # args.max_target_length
        return_dict_in_generate=True,
        output_scores=True)
#     print('outs =', outs)

    dec = [tokenizer.decode(ids, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False) 
                for ids in outs.sequences]
#     conf = [s.cpu().item() for s in torch.exp(outs.scores)]
#     target = [tokenizer.decode(ids, skip_special_tokens=True, 
#                                clean_up_tokenization_spaces=False) 
#                 for ids in batch["target_ids"]]

    outputs.extend(dec)
#     confidences.extend(conf)
#     targets.extend(target)
    
with open('/kaggle/working/results.txt', 'w', encoding='utf-8') as f:
    for pred in outputs:
        f.write(pred + '\n')
