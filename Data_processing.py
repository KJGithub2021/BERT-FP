from transformers import BertTokenizer
import pickle
import csv
from tqdm import tqdm

def build_response_dict(filepath):
    response_dict = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            response_id = int(row[0])
            response_dict[response_id] = row[1]  # Context used as response
    return response_dict
    

def FT_data(file_path, response_dict, enforce=0, tokenizer=None):
    y = []
    cr_list = []  # list of combined context and response
    fbase = file_path.split("/")[-1]
    skipCount = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        lineno = 0
        for row in reader:
            #print("Row: ", row)
            lineno += 1
            utt = row[1]
            #print("utt: ", utt)
            valids = row[2]
            #print("valids: ", valids)
            invalids = row[3]
            #print("invalids: ", invalids)
            isValid = valids != "NA"
            splits = [[], []]
            if valids != "NA":
                splits[0] = [int(x) for x in valids.split("|")]
                #print("splits[0]: ", splits[0])
            if invalids != "NA":
                splits[1] = [int(x) for x in invalids.split("|")]
                #print("splits[1]: ", splits[1])
            respCount = len(splits[0]) + len(splits[1])
            if enforce != 0 and respCount != enforce:
                skipCount += 1
                continue
            if(lineno%10000 == 0): 
                print("Splits: ", splits)


            for label in range(len(splits)):
                for id in splits[label]:
                    y.append((label + 1) % 2)
                    #print("Label: ", label, " ID: ", id)

                    # Tokenize the combined context and response using BERT tokenizer
                    context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt))
                    #print("context_tokens: ",  context_tokens)
                    response_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_dict[id]))
                    #print("response_tokens: ", response_tokens )
                    # Combine context and response
                    cr_tokens = context_tokens + [tokenizer.sep_token_id] + response_tokens
                    #print("cr_tokens", cr_tokens)
                    cr_list.append(cr_tokens)
    if enforce != 0:
        print("{} Records skipped due to enforce={} is = {}".format(fbase, enforce, skipCount))
    return y, cr_list




if __name__ == '__main__':
    #Fine_tuning data constuction
    #including tokenization step
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased",do_lower_case=True)
    special_tokens_dict = {'additional_special_tokens': ['__eot__', '__eou__']}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)
    
    response_dict = build_response_dict('ubuntu_data/responses.txt')

    train, test, dev = {}, {}, {}
    train['y'], train['cr'] = FT_data('ubuntu_data/train.txt', response_dict, tokenizer=bert_tokenizer)
    dev['y'], dev['cr'] = FT_data('ubuntu_data/valid.txt',  response_dict, tokenizer=bert_tokenizer)
    test['y'], test['cr']= FT_data('ubuntu_data/test.txt', response_dict, tokenizer=bert_tokenizer)
    #char_vocab = defaultdict(float)
    dataset = train, dev, test
    pickle.dump(dataset, open('ubuntu_data/ubuntu_dataset_1M.pkl', 'wb'))
