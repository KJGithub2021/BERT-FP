print('Calculating ROUGE!\n')
import evaluate
import time

rouge = evaluate.load('rouge')

def load_responses(fname):
    start_time = time.time()  # Talha, 9/2/24

    responses = {}
    with open(fname, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            fields = line.split('\t')
            if len(fields) != 2:
                print("WRONG LINE: {}".format(line))
                responses[fields[0]] = 'UNKNOWN'
            else:
                responses[fields[0]] = fields[1]

    print("load_responses took --- %s seconds ---" % (time.time() - start_time))
    return responses

def load_responseids(test_file):
    with open(test_file, 'rt', encoding='utf-8') as file:
        data = file.readlines()

    response_ids = []
    for line in data:
        parts = line.strip().split('\t')
        reference_id = parts[2]
        negative_ids = parts[3].split('|')
        
        # Skip if the total number of response IDs is less than 2 (no valid negative examples)
        if len(negative_ids) < 1:
            continue
        
        response_ids.append((reference_id, negative_ids))
    
    return response_ids

# Find the Prediction ID from scores.txt
def find_prediction_id(scores_file, response_ids):
    with open(scores_file, 'r', encoding='utf-8') as file:
        scores = file.readlines()

    prediction_ids = []
    response_count = 0 

    for response_set in response_ids:
        num_responses = len(response_set[1]) + 1  # +1 for the reference response
        
        # Skip if there aren't enough scores for this response set
        if response_count + num_responses > len(scores):
            break

        group_scores = [float(score.split()[0]) for score in scores[response_count:response_count + num_responses]]
        max_index = group_scores.index(max(group_scores))

        if max_index == 0:
            prediction_id = response_set[0]
        else: 
            prediction_id = response_set[1][max_index-1]  # max_index corresponds to the negative IDs

        prediction_ids.append(prediction_id)
        response_count += num_responses  # Update the count of responses covered

    return prediction_ids

# Calculate ROUGE
def calculate_rouge(response_data, response_ids, scores_file):
    # Find prediction IDs
    prediction_ids = find_prediction_id(scores_file, response_ids)
    
    references = []
    predictions = []
    
    # Fetch the texts for the reference and prediction IDs
    for (reference_id, _), prediction_id in zip(response_ids, prediction_ids):
        reference_text = response_data.get(reference_id, "UNKNOWN")
        prediction_text = response_data.get(prediction_id, "UNKNOWN")
        references.append(reference_text)
        predictions.append(prediction_text)
    
    # Calculate ROUGE
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

# Load responses and response IDs
response_data = load_responses('ubuntu_data/responses.txt')
response_ids = load_responseids('ubuntu_data/test.txt')

# Calculate ROUGE scores
scores = calculate_rouge(response_data, response_ids, 'Fine-Tuning/scorefile.txt')
print("ROUGE Scores:", scores)
