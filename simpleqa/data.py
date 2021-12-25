from tqdm import tqdm
import json

def find_loc_approx(text, target, tokenizer):
    loc = text.find(target)
    if loc == -1:
        return -1
    return len(tokenizer.encode(text[:loc], add_special_tokens =False))

def get_input(text, target, loc, max_len = 384, doc_stride = 128):
    input_ids = []
    masks = []
    starts = []
    ends = []
    for i in range(0, len(text), doc_stride):
        if i <= loc and loc + len(target) <= min(i+max_len, len(text)):
            ntok = min(i+max_len, len(text)) - i
            ids = text[i:min(i+max_len, len(text))] + [0]*(max_len-ntok)
            input_ids.append(ids)
            masks.append([1]*ntok + [0]*(max_len-ntok))
            starts.append(loc - i)
            ends.append(loc - i + len(target))
    return input_ids, masks, starts, ends

def read_data(file, target_name, tokenizer, max_len = 384, doc_stride = 128):
    data = json.load(open(file, encoding='utf-8'))
    texts = [item['text'] for item in data]
    targets = [item[target_name] for item in data]

    text_tok = [tokenizer.encode(text, add_special_tokens =False) for text in texts]
    target_tok = [tokenizer.encode(text, add_special_tokens =False) for text in targets]
    target_loc = []
    for i, text, target in zip(range(len(texts)), texts, targets):
        target_loc.append(find_loc_approx(text, target, tokenizer))
        
    all_input_ids, all_masks, all_starts, all_ends  = [], [], [], []
    sample_loc = [0]
    for i in tqdm(range(len(text_tok))):
        input_ids, masks, starts, ends = get_input(text_tok[i], target_tok[i], target_loc[i], 
                                                   max_len = max_len, doc_stride = doc_stride)
        all_input_ids.extend(input_ids)
        all_masks.extend(masks)
        all_starts.extend(starts)
        all_ends.extend(ends)
        sample_loc.append(sample_loc[-1] + len(input_ids))
    
    return all_input_ids, all_masks, all_starts, all_ends, sample_loc

def get_input_eval(text, max_len = 384, doc_stride = 128):
    input_ids = []
    masks = []
    starts = []
    ends = []
    for i in range(0, len(text), doc_stride):
        ntok = min(i+max_len, len(text)) - i
        ids = text[i:min(i+max_len, len(text))] + [0]*(max_len-ntok)
        input_ids.append(ids)
        masks.append([1]*ntok + [0]*(max_len-ntok))
        starts.append(i) # dummy
        ends.append(i) # dummy
    return input_ids, masks, starts, ends
    
def read_data_eval(file, target_name, tokenizer, max_len = 384, doc_stride = 128):
    data = json.load(open(file, encoding='utf-8'))
    texts = [item['text'] for item in data]
    targets = [item[target_name] for item in data]

    text_tok = [tokenizer.encode(text, add_special_tokens =False) for text in texts]
        
    all_input_ids, all_masks, all_starts, all_ends  = [], [], [], []
    sample_loc = [0]
    for i in tqdm(range(len(text_tok))):
        input_ids, masks, starts, ends = get_input_eval(text_tok[i], 
                                                   max_len = max_len, doc_stride = doc_stride)
        all_input_ids.extend(input_ids)
        all_masks.extend(masks)
        all_starts.extend(starts)
        all_ends.extend(ends)
        sample_loc.append(sample_loc[-1] + len(input_ids))
    
    return all_input_ids, all_masks, all_starts, all_ends, sample_loc


