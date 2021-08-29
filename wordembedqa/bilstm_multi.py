import pickle 
from flair.data import Sentence
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import collections
from wordembed_models import WordEmbedBiLSTMMultiQA, MAX_LEN
from wordembed_scores import compute_predictions_logits
import sys

np.random.seed(42) 
torch.manual_seed(42)

####
w2v_texts, vectors, hazard_locs, state_locs, effect_locs = pickle.load(open('wordembed.pkl', 'rb'))
vectors = [x[:MAX_LEN,:] for x in vectors]

####
n = len(w2v_texts)
ids = np.random.permutation(n)  # random permutation
ids = np.concatenate((ids, ids)) # repeat 2 times

n_cv = 10 # 5, 10
all_test_ids = []
all_train_ids = []
for i in range(n_cv):
    #
    all_test_ids.append(ids[i*n//n_cv : (i+1)*n//n_cv])
    all_train_ids.append(ids[(i+1)*n//n_cv : i*n//n_cv + n])

batch_size = int(sys.argv[1])
learning_rate = float(sys.argv[2])
epochs = int(sys.argv[3])
max_answer_length = int(sys.argv[4])
print('batch_size =', batch_size)
print('learning_rate =', learning_rate)
print('epochs =', epochs)
print('max_answer_length =', max_answer_length)

for fold in range(10):
    train_vectors = [vectors[i] for i in all_train_ids[fold]]    
    test_vectors = [vectors[i] for i in all_test_ids[fold]]

    train_locs = [(hazard_locs[i][0], hazard_locs[i][1], state_locs[i][0], state_locs[i][1],
               effect_locs[i][0], effect_locs[i][1]) for i in all_train_ids[fold]]    
    test_locs = [(hazard_locs[i][0], hazard_locs[i][1], state_locs[i][0], state_locs[i][1],
               effect_locs[i][0], effect_locs[i][1]) for i in all_test_ids[fold]]

    train_texts = [w2v_texts[i] for i in all_train_ids[fold]]  
    test_texts = [w2v_texts[i] for i in all_test_ids[fold]]  
    train_num_tokens = [len(text.tokens) for text in train_texts]
    test_num_tokens = [len(text.tokens) for text in test_texts]

    train_data = TensorDataset(torch.tensor(train_vectors, dtype=torch.float),
                               torch.tensor(train_locs, dtype=torch.int64),
                               torch.tensor(train_num_tokens, dtype=torch.int64)
                              )
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    eval_data = TensorDataset(torch.tensor(test_vectors, dtype=torch.float),
                               torch.tensor(test_locs, dtype=torch.int64),
                               torch.tensor(test_num_tokens, dtype=torch.int64)
                              )
    eval_sampler = SequentialSampler(eval_data)
    validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    
    #### TRAIN
    model = WordEmbedBiLSTMMultiQA(300, 20)

    print(model)
    model = model.cuda()

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for epoch in range(epochs):  # MLP (100), BiLSTM (20, 10)
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t.cuda() for t in batch)

            x, all_positions, ignore_index = batch
            optimizer.zero_grad()
            loss, _ = model(x, all_positions) #, ignore_index)

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1

        print('Training loss =%.4f' % (tr_loss / nb_tr_steps))
    
    #### EVAL
    model.eval()

    cid = 0
    count = 0
    all_list = [] # hiepnh
    total_loss = 0.0

    all_results = []
    Result = collections.namedtuple("Result", ["all_logits", "num_tokens"])

    for batch in validation_data_loader:
        batch = tuple(t.cuda() for t in batch)
        x, all_positions, num_tokens = batch
        
        with torch.no_grad():
            loss, all_logits = model(x, all_positions)
            all_pred = all_logits.detach().cpu().numpy()
            all_list.append(all_pred)

        for idx, (pred, n_tokens) in enumerate(zip(all_pred, num_tokens)):
            pred_id = np.argmax(pred)
            loss_val = loss.detach().cpu().numpy()
    #         print('-- cid =', cid, 'loss =', loss_val)
    #         print('true', test_starts[cid], test_ends[cid])
    #         print('pred', start_id, end_id)
            cid += 1
            total_loss += loss_val
            #
            all_results.append(Result(all_logits=pred, num_tokens=n_tokens.cpu().numpy()))
    print('eval loss =', total_loss/cid)
    
    #### POST-PROCESS
    test_tokens = []
    for sample_text in test_texts:
        test_tokens.append([token.text for token in sample_text.tokens])

    pickle.dump((test_tokens, all_results), open('temp/' + str(fold) + '.pkl', 'wb'))

