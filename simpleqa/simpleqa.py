import json
import glob
import logging
import os
import random
import timeit
import argparse
import sys
import collections
from colorama import Fore
import numpy as np
from tqdm import tqdm, trange
import pickle

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, T5Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

from model import SimpleBertQA
from data import read_data, read_data_eval
from score import compute_predictions_logits, compute_scores


## TRAIN
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
def train(args, train_dataset, model, tokenizer, eval_input_ids, eval_data, eval_loc):
    """ Train the model """
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Instantaneous batch size = %d", args.train_batch_size)
    print(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for epoch in train_iterator:
        training_pbar = tqdm(total=len(train_dataset),
                         position=0, leave=True,
                         file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[2],
                "end_positions": batch[3],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            training_pbar.update(batch[0].size(0)) # hiepnh
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print("lr", scheduler.get_lr()[0])
                    print("loss", (tr_loss - logging_loss) / args.logging_steps)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                training_pbar.close() # hiepnh
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            
        ## EVAL
        evaluate(args, eval_input_ids, eval_data, model, tokenizer, prefix=str(epoch))

        p,r,f1 = compute_scores(args.eval_file, eval_loc, 
                                args.output_dir + '/nbest_predictions_' + str(epoch) + '.json', args.target)
        print('%.3f'%p, '%.3f'%r, '%.3f'%f1) 

    return global_step, tr_loss / global_step

## EVAL
Result = collections.namedtuple("Result", ["start_logits", "end_logits", "num_tokens"])

def evaluate(args, eval_input_ids, dataset, model, tokenizer, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(dataset))
    print("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    eval_pbar = tqdm(total=len(dataset),
                     position=0, leave=True,
                     file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[2],
                "end_positions": batch[3],
            }

            outputs = model(**inputs)
            # print('outputs =', outputs)
            for start_logits, end_logits, mask in zip(outputs[1], outputs[2], batch[1]):
                start, end = start_logits.detach().cpu().numpy(), end_logits.detach().cpu().numpy()
                n_tokens = mask.cpu().numpy().sum()

                all_results.append(Result(start_logits=start, end_logits=end, num_tokens=n_tokens))

        eval_pbar.update(batch[0].size(0)) # hiepnh
    eval_pbar.close() # hiepnh

    print('len(all_results) =', len(all_results))

    evalTime = timeit.default_timer() - start_time
    print("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    predictions = compute_predictions_logits(
        eval_input_ids,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        tokenizer,
        output_prediction_file,
        output_nbest_file
    )
    json.dump(eval_loc, open(os.path.join(args.output_dir, "eval_loc.json"), "w"))
    pickle.dump(all_results, open(os.path.join(args.output_dir, "all_results_" + prefix + ".pkl"), "wb"))

## MAIN
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default='TurkuNLP/wikibert-base-ja-cased', type=str, required=True, help="Path to pretrained model", )
parser.add_argument("--device", default='cuda:0', type=str, required=False, help="device", )
parser.add_argument("--do_train", dest='do_train', action='store_true', help="do_train", )
parser.add_argument("--do_eval", dest='do_eval', action='store_true', help="do_eval", )
parser.add_argument("--train_file", default='train_0.json', type=str, required=True, help="train_file", )
parser.add_argument("--eval_file", default='test_0.json', type=str, required=True, help="eval_file", )
parser.add_argument("--target", default='hazard', type=str, required=True, help="target", )
parser.add_argument("--max_len", default=384, type=int, required=False, help="max_len", )
parser.add_argument("--doc_stride", default=128, type=int, required=False, help="doc_stride", )
parser.add_argument("--n_best_size", default=10, type=int, required=False, help="n_best_size", )
parser.add_argument("--max_answer_length", default=30, type=int, required=False, help="max_answer_length", )

parser.add_argument("--learning_rate", default=3e-4, type=float, required=False, help="learning_rate", )
parser.add_argument("--seed", default=42, type=int, required=False, help="seed", )
parser.add_argument("--train_batch_size", default=8, type=int, required=False, help="train_batch_size", )
parser.add_argument("--eval_batch_size", default=8, type=int, required=False, help="eval_batch_size", )
parser.add_argument("--num_train_epochs", default=5, type=int, required=False, help="num_train_epochs", )
parser.add_argument("--max_steps", default=-1, type=int, required=False, help="max_steps", )
parser.add_argument("--gradient_accumulation_steps", default=1, type=int, required=False, help="gradient_accumulation_steps", )
parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False, help="adam_epsilon", )
parser.add_argument("--warmup_steps", default=0, type=int, required=False, help="warmup_steps", )
parser.add_argument("--local_rank", default=-1, type=int, required=False, help="local_rank", )
parser.add_argument("--weight_decay", default=0.0, type=float, required=False, help="weight_decay", )
parser.add_argument("--max_grad_norm", default=1.0, type=float, required=False, help="max_grad_norm", )
parser.add_argument("--logging_steps", default=500, type=int, required=False, help="logging_steps", )
parser.add_argument("--output_dir", default='output', type=str, required=True, help="output_dir", )

args = parser.parse_args()

print(args)

set_seed(args)

if 'rinna' in args.model_name_or_path:
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

eval_input_ids, eval_masks, eval_starts, eval_ends, eval_loc = \
    read_data_eval(args.eval_file, args.target, tokenizer, max_len=args.max_len, doc_stride=args.doc_stride)
eval_data = TensorDataset(torch.tensor(eval_input_ids, dtype=torch.int64),
                    torch.tensor(eval_masks, dtype=torch.int64),
                    torch.tensor(eval_starts, dtype=torch.int64),
                    torch.tensor(eval_ends, dtype=torch.int64)
                    )
    
## TRAIN
if args.do_train:
    model = SimpleBertQA.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    train_input_ids, train_masks, train_starts, train_ends, _ = \
            read_data(args.train_file, args.target, tokenizer, max_len=args.max_len, doc_stride=args.doc_stride)
    train_data = TensorDataset(torch.tensor(train_input_ids, dtype=torch.int64),
                            torch.tensor(train_masks, dtype=torch.int64),
                            torch.tensor(train_starts, dtype=torch.int64),
                            torch.tensor(train_ends, dtype=torch.int64)
                            )
    final_step, final_loss = train(args, train_data, model, tokenizer, eval_input_ids, eval_data, eval_loc) # eval each ep
#     final_step, final_loss = train(args, train_data, model, tokenizer)

    print('final_step =', final_step, 'final_loss =', final_loss)

    print("Saving model checkpoint to %s", args.output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

## EVAL
if args.do_eval:
#     eval_input_ids, eval_masks, eval_starts, eval_ends, eval_loc = \
#         read_data(args.eval_file, args.target, tokenizer, max_len=args.max_len, doc_stride=args.doc_stride)
#     eval_data = TensorDataset(torch.tensor(eval_input_ids, dtype=torch.int64),
#                             torch.tensor(eval_masks, dtype=torch.int64),
#                             torch.tensor(eval_starts, dtype=torch.int64),
#                             torch.tensor(eval_ends, dtype=torch.int64)
#                             )
    if args.do_train:
        model = SimpleBertQA.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    else:
        model = SimpleBertQA.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    
#     evaluate(args, eval_input_ids, eval_data, model, tokenizer)
    
#     p,r,f1 = compute_scores(args.eval_file, eval_loc, args.output_dir + '/nbest_predictions_.json', args.target)
#     print('%.3f'%p, '%.3f'%r, '%.3f'%f1)   