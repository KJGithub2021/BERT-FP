from transformers import BertTokenizer
import pickle
from tqdm import tqdm
import csv

def build_response_dict(filepath):
    response_dict = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            response_id = row[0]
            response_text = row[1]  
            response_text = response_text.replace('__eou__', '').strip()
            response_dict[response_id] = response_text
    return response_dict

def convert_v2_to_v1_train(v2_filepath, response_dict):
    v1_lines = []
    with open(v2_filepath, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) < 4:
                continue
            context = row[1]
            pos_response_id = row[2]
            neg_response_id = row[3]
            if pos_response_id == 'NA':
                start = 0
                response_id = neg_response_id
            else:
                start = 1
                response_id = pos_response_id
            response_text = response_dict.get(response_id, 'UNKNOWN_RESPONSE')
            context = context.replace('__eot__', '').strip()
            context_parts = context.split("__eou__")
            tab_separated_context = "\t".join([part.strip() for part in context_parts if part.strip()])
            v1_line = f"{start}\t{tab_separated_context}\t{response_text}"
            v1_lines.append(v1_line)
    return v1_lines

def convert_v2_to_v1_test(v2_filepath, response_dict):
    v1_lines = []
    with open(v2_filepath, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) < 4:
                continue
            context = row[1]
            pos_response_id = row[2]
            neg_response_ids = row[3].split('|')
            context = context.replace('__eot__', '').strip()
            context_parts = context.split("__eou__")
            tab_separated_context = "\t".join([part.strip() for part in context_parts if part.strip()])
            pos_response_text = response_dict.get(pos_response_id, 'UNKNOWN_RESPONSE')
            v1_line_pos = f"1\t{tab_separated_context}\t{pos_response_text}"
            v1_lines.append(v1_line_pos)
            for neg_response_id in neg_response_ids:
                neg_response_text = response_dict.get(neg_response_id, 'UNKNOWN_RESPONSE')
                v1_line_neg = f"0\t{tab_separated_context}\t{neg_response_text}"
                v1_lines.append(v1_line_neg)
    return v1_lines

def write_v1_output(v1_lines, output_filepath):
    with open(output_filepath, 'w') as file:
        for line in v1_lines:
            file.write(line + '\n')

def FT_data(file, tokenizer=None):
    print("Starting Fine-tuning data construction: ", file)
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    y = [int(a[0]) for a in data]
    cr = [[sen for sen in a[1:]] for a in data]
    cr_list = []
    cnt = 0
    for s in tqdm(cr):
        s_list = []
        for sen in s[:-1]:
            if len(sen) == 0:
                cnt += 1
                continue
            s_list += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen + tokenizer.eos_token))
        s_list = s_list + [tokenizer.sep_token_id]
        s_list += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s[-1]))
        cr_list.append(s_list)

    print("Fine-tuning data construction complete")
    return y, cr_list

def PT_data(file):
    print("Starting Post Training data construction: ", file)
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    y = [int(a[0]) for a in data]
    cr = [[sen for sen in a[1:]] for a in data]
    crnew = []
    for i, crsingle in enumerate(cr):
        if y[i] == 1:
            crnew.append(crsingle)
    pickle.dump(crnew, file=open("ubuntu_data/ubuntu_post_train.pkl", 'wb'))
    print("Post Training data construction complete")

if __name__ == '__main__':
    # Convert v2 files to v1 format
    print("Structuring Input Data")
    response_dict = build_response_dict('ubuntu_data/responses.txt')
    v1_train_lines = convert_v2_to_v1_train('ubuntu_data/train.txt', response_dict)
    write_v1_output(v1_train_lines, 'ubuntu_data/v1_train.txt')
    
    v1_valid_lines = convert_v2_to_v1_test('ubuntu_data/valid.txt', response_dict)
    write_v1_output(v1_valid_lines, 'ubuntu_data/v1_valid.txt')
    
    v1_test_lines = convert_v2_to_v1_test('ubuntu_data/test.txt', response_dict)
    write_v1_output(v1_test_lines, 'ubuntu_data/v1_test.txt')

    # Fine-tuning data construction
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    special_tokens_dict = {'eos_token': '[eos]'}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)

    train, test, dev = {}, {}, {}
    train['y'], train['cr'] = FT_data('ubuntu_data/v1_train.txt', tokenizer=bert_tokenizer)
    dev['y'], dev['cr'] = FT_data('ubuntu_data/v1_valid.txt', tokenizer=bert_tokenizer)
    test['y'], test['cr'] = FT_data('ubuntu_data/v1_test.txt', tokenizer=bert_tokenizer)
    
    dataset = train, dev, test
    pickle.dump(dataset, open('ubuntu_data/ubuntu_dataset_1M.pkl', 'wb'))

    # Post-training data construction
    PT_data('ubuntu_data/v1_train.txt')
