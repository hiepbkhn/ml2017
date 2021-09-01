import pickle 
from flair.data import Sentence
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import collections
import sys

np.random.seed(42) 
torch.manual_seed(42)

def prepare_loc_probs(locs, max_len):
    starts = np.zeros((len(locs),max_len))
    ends = np.zeros((len(locs),max_len))
    for i, ca_locs in enumerate(locs):
        n_ca = len(ca_locs)
        for x in ca_locs:
            starts[i][x[0]] = 1.0/n_ca
            ends[i][x[1]] = 1.0/n_ca
    return starts, ends    

##
MAX_LEN = 512
w2v_texts, vectors, hazard_locs, state_locs, effect_locs = \
                pickle.load(open('wordembed-626.pkl', 'rb'))
temp = []
for vec in vectors:
    if vec.shape[0] <= MAX_LEN:
        pad = np.zeros((MAX_LEN, 300))
        pad[:vec.shape[0],:] = vec 
        temp.append(pad)
    else:
        temp.append(vec[:MAX_LEN,:])
vectors = temp

####
n = len(w2v_texts)
ids = np.random.permutation(n)  # random permutation
ids = np.concatenate((ids, ids)) # repeat 2 times

n_cv = 5 # 5, 10
all_test_ids = []
all_train_ids = []
for i in range(n_cv):
    #
    all_test_ids.append(ids[i*n//n_cv : (i+1)*n//n_cv])
    all_train_ids.append(ids[(i+1)*n//n_cv : i*n//n_cv + n])

####
if sys.argv[1] == 'hazard':
    locs = hazard_locs
if sys.argv[1] == 'state':
    locs = state_locs
if sys.argv[1] == 'effect':
    locs = effect_locs        

learning_rate = float(sys.argv[2])
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
print('learning_rate =', learning_rate)
print('epochs =', epochs)
print('batch_size =', batch_size)

for fold in range(n_cv):
    train_vectors = [vectors[i] for i in all_train_ids[fold]]    
    test_vectors = [vectors[i] for i in all_test_ids[fold]]

    train_locs = [locs[i] for i in all_train_ids[fold]]    
    test_locs = [locs[i] for i in all_test_ids[fold]]
        
    train_starts, train_ends = prepare_loc_probs(train_locs, MAX_LEN)
    test_starts, test_ends = prepare_loc_probs(test_locs, MAX_LEN)

    train_texts = [w2v_texts[i] for i in all_train_ids[fold]]  
    test_texts = [w2v_texts[i] for i in all_test_ids[fold]]  
    train_num_tokens = [len(text.tokens) for text in train_texts]
    test_num_tokens = [len(text.tokens) for text in test_texts]

    
    train_data = TensorDataset(torch.tensor(train_vectors, dtype=torch.float),
                               torch.tensor(train_starts, dtype=torch.float),
                               torch.tensor(train_ends, dtype=torch.float),
                               torch.tensor(train_num_tokens, dtype=torch.int64)
                              )
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    eval_data = TensorDataset(torch.tensor(test_vectors, dtype=torch.float),
                               torch.tensor(test_starts, dtype=torch.float),
                               torch.tensor(test_ends, dtype=torch.float),
                               torch.tensor(test_num_tokens, dtype=torch.int64)
                              )
    eval_sampler = SequentialSampler(eval_data)
    validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    
    #### TRAIN
    model = WordEmbedUNetQA()

    if fold == 0:
        print(model)
    model = model.cuda()

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for epoch in range(epochs):
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t.cuda() for t in batch)

            x, start_positions, end_positions, ignore_index = batch
            optimizer.zero_grad()
            loss, _, _ = model(torch.transpose(x, 1, 2), start_positions, end_positions) #, ignore_index)

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1

        print('Training loss =%.4f' % (tr_loss / nb_tr_steps))
    
    #### EVAL
    model.eval()

    cid = 0
    count = 0
    start_list = [] # hiepnh
    end_list = [] 
    total_loss = 0.0

    all_results = []
    Result = collections.namedtuple("Result", ["start_logits", "end_logits", "num_tokens"])

    for batch in validation_data_loader:
        batch = tuple(t.cuda() for t in batch)
        x, start_positions, end_positions, num_tokens = batch
        if isinstance(model, WordEmbedUNetQA):
            x = torch.transpose(x, 1, 2)
        
        with torch.no_grad():
            loss, start_logits, end_logits = model(x, start_positions, end_positions)
            pred_start, pred_end = start_logits.detach().cpu().numpy(), end_logits.detach().cpu().numpy()
            start_list.append(pred_start)
            end_list.append(pred_end)

        for idx, (start, end, n_tokens) in enumerate(zip(pred_start, pred_end, num_tokens)):
            start_id = np.argmax(start)
            end_id = np.argmax(end)
            loss_val = loss.detach().cpu().numpy()
    #         print('-- cid =', cid, 'loss =', loss_val)
    #         print('true', test_starts[cid], test_ends[cid])
    #         print('pred', start_id, end_id)
            cid += 1
            total_loss += loss_val
            #
            all_results.append(Result(start_logits=start, end_logits=end, num_tokens=n_tokens.cpu().numpy()))
    print('eval loss =', total_loss/cid)
    
    #### POST-PROCESS
    test_tokens = []
    for sample_text in test_texts:
        test_tokens.append([token.text for token in sample_text.tokens])

    pickle.dump((test_tokens, all_results), open('temp/' + str(fold) + '.pkl', 'wb'))

    # ret = compute_predictions_logits(
    #     test_tokens,
    #     all_results,
    #     15,
    #     max_answer_length,
    #     'temp/prediction-' + str(fold) + '.json',
    #     'temp/nbest-' + str(fold) + '.json',
    # )