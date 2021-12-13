import math
import json
import collections
import pylcs
import numpy as np

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def _get_best_indexes(logits, num_tokens, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits[:num_tokens]), key=lambda x: x[1], reverse=True)
    
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def compute_predictions_logits(
    all_input_ids,
    all_results,
    n_best_size,
    max_answer_length,
    tokenizer,
    output_prediction_file,
    output_nbest_file
):

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for idx, result in enumerate(all_results):
        prelim_predictions = []
        start_indexes = _get_best_indexes(result.start_logits, result.num_tokens, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, result.num_tokens, n_best_size)

        # if we could have irrelevant answers, get the min score of irrelevant
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= result.num_tokens:
                    continue
                if end_index >= result.num_tokens:
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index],
                    )
                )
                
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction
                final_text = tokenizer.decode(all_input_ids[idx][pred.start_index:pred.end_index]) # +1
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ''
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit, start_index=pred.start_index, end_index=pred.end_index))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        all_predictions[idx] = best_non_null_entry.text if best_non_null_entry is not None else ''

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = str(probs[i]) # TypeError: Object of type float32 is not JSON serializable
            output["start_logit"] = str(entry.start_logit)
            output["end_logit"] = str(entry.end_logit)
            output["start_index"] = entry.start_index
            output["end_index"] = entry.end_index
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"

        all_nbest_json[idx] = nbest_json

    if output_prediction_file:
        with open(output_prediction_file, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")
    if output_nbest_file:
        with open(output_nbest_file, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")
            
    return all_predictions

def compute_acc_list(true_list, pred_list):
    assert len(true_list) == len(pred_list)
    p_list, r_list, f1_list = [], [], []
    for x, y in zip(true_list, pred_list):
        val = pylcs.lcs(x, y)
        p = float(val)/len(y) if len(y) > 0 else 0.0
        r = float(val)/len(x) if len(x) > 0 else 0.0
        f1 = 2*p*r/(p+r) if val > 0 else 0.0
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        
    return p_list, r_list, f1_list    

def compute_scores(eval_file, eval_loc, pred_file, target):

    data = json.load(open(eval_file, encoding='utf-8'))
    ca_target = [item[target] for item in data]

    pred = json.load(open(pred_file, encoding='utf-8'))
    pred = list(pred.values())
    pred_texts = [item[0]['text'] for item in pred]
    pred_probs = [float(item[0]['probability']) for item in pred]

    pred_target = []
    for i in range(len(ca_target)):
        best, prob = '', 0.0
        for j in range(eval_loc[i], eval_loc[i+1]):
            if prob < pred_probs[j]:
                best = pred_texts[j]
                prob = pred_probs[j]
        pred_target.append(best.replace(' ', ''))
            
    target_p, target_r, target_f1 = compute_acc_list(ca_target, pred_target)    
    return np.mean(target_p), np.mean(target_r), np.mean(target_f1)  