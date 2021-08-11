import pickle 
from flair.data import Sentence
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import collections
from wordembed_models import WordEmbedQA, WordEmbedLSTMQA, WordEmbedBiLSTMQA, WordEmbedTransformerQA, WordEmbedUNetQA
from wordembed_scores import compute_predictions_logits

np.random.seed(42) 

####
w2v_texts, vectors, hazard_locs, state_locs, effect_locs = pickle.load(open('wordembed.pkl', 'rb'))
vectors = [x[:900,:] for x in vectors]

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

####
locs = state_locs

for fold in range(10):
    train_vectors = [vectors[i] for i in all_train_ids[fold]]    
    test_vectors = [vectors[i] for i in all_test_ids[fold]]

    train_locs = [locs[i] for i in all_train_ids[fold]]    
    test_locs = [locs[i] for i in all_test_ids[fold]]
        
    train_starts = [x[0] for x in train_locs]
    train_ends = [x[1] for x in train_locs]

    test_starts = [x[0] for x in test_locs]
    test_ends = [x[1] for x in test_locs]

    train_texts = [w2v_texts[i] for i in all_train_ids[fold]]  
    test_texts = [w2v_texts[i] for i in all_test_ids[fold]]  
    train_num_tokens = [len(text.tokens) for text in train_texts]
    test_num_tokens = [len(text.tokens) for text in test_texts]

    
    batch_size = 8

    train_data = TensorDataset(torch.tensor(train_vectors, dtype=torch.float),
                               torch.tensor(train_starts, dtype=torch.int64),
                               torch.tensor(train_ends, dtype=torch.int64),
                               torch.tensor(train_num_tokens, dtype=torch.int64)
                              )
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    eval_data = TensorDataset(torch.tensor(test_vectors, dtype=torch.float),
                               torch.tensor(test_starts, dtype=torch.int64),
                               torch.tensor(test_ends, dtype=torch.int64),
                               torch.tensor(test_num_tokens, dtype=torch.int64)
                              )
    eval_sampler = SequentialSampler(eval_data)
    validation_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
    
    #### TRAIN
    model = WordEmbedTransformerQA()

    print(model)
    model = model.cuda()

    learning_rate = 0.00001 # 0.001, 0.0001, 0.00001
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for epoch in range(100):
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t.cuda() for t in batch)

            x, start_positions, end_positions, ignore_index = batch
            optimizer.zero_grad()
            loss, _, _ = model(x, start_positions, end_positions) #, ignore_index)

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
            all_results.append(Result(start_logits=start, end_logits=end, num_tokens=n_tokens))
    print('eval loss =', total_loss/cid)
    
    #### POST-PROCESS
    test_tokens = []
    for sample_text in test_texts:
        test_tokens.append([token.text for token in sample_text.tokens])
    ret = compute_predictions_logits(
        test_tokens,
        all_results,
        15,
        100,
        'temp/prediction-' + str(fold) + '.json',
        'temp/nbest-' + str(fold) + '.json',
    )