import math
import json
import collections

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
    all_examples,
    all_results,
    n_best_size,
    max_answer_length,
    output_prediction_file,
    output_nbest_file,
):

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for idx, (example, result) in enumerate(zip(all_examples, all_results)):
        prelim_predictions = []
        start_indexes = _get_best_indexes(result.start_logits, result.num_tokens, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, result.num_tokens, n_best_size)

        # if we could have irrelevant answers, get the min score of irrelevant
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(example):
                    continue
                if end_index >= len(example):
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
                final_text = ''.join(all_examples[idx][pred.start_index:pred.end_index+1])
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
        all_predictions[idx] = best_non_null_entry.text

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