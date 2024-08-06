from transformers import BertTokenizer
import pickle
import csv
from tqdm import tqdm

def OLDFT_data(file, tokenizer=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    print(data)
    y = [int(a[0]) for a in data]
    cr = [ [sen for sen in a[1:]] for a in data]
    cr_list=[]
    cnt=0
    for s in tqdm(cr):
        s_list=[]
        for sen in s[:-1]:
            if len(sen)==0:
                cnt+=1
                continue
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+tokenizer.eos_token))
        s_list=s_list+[tokenizer.sep_token_id]
        s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s[-1]))
        cr_list.append(s_list)
    print(cnt)
    return y, cr_list

def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r') as file:
        for line in file:
            word, index = line.strip().split('\t')
            vocab[word] = int(index)
    return vocab

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
    
def tokenize(vocab, utt):
    return [vocab.get(tok, vocab["UNKNOWN"]) for tok in utt.split()]

def FT_data(file_path, vocab, response_dict, enforce=0, tokenizer=None):
    y = []
    cr_list = []  # list of combined context and response
    fbase = file_path.split("/")[-1]
    skipCount = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        lineno = 0
        for row in reader:
            print("Row: ", row)
            lineno += 1
            utt = row[1]
            print("utt: ", utt)
            valids = row[2]
            print("valids: ", valids)
            invalids = row[3]
            print("invalids: ", invalids)
            isValid = valids != "NA"
            splits = [[], []]
            if valids != "NA":
                splits[0] = [int(x) for x in valids.split("|")]
                print("splits[0]: ", splits[0])
            if invalids != "NA":
                splits[1] = [int(x) for x in invalids.split("|")]
                print("splits[1]: ", splits[1])
            respCount = len(splits[0]) + len(splits[1])
            if enforce != 0 and respCount != enforce:
                skipCount += 1
                continue
            print("Splits: ", splits)


            for label in range(len(splits)):
                for id in splits[label]:
                    y.append((label + 1) % 2)
                    print("Label: ", label, " ID: ", id)

                    # Tokenize the combined context and response using BERT tokenizer
                    context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt))
                    print("context_tokens: ",  context_tokens)
                    response_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_dict[id]))
                    print("response_tokens: ", response_tokens )
                    # Combine context and response
                    cr_tokens = context_tokens + [tokenizer.sep_token_id] + response_tokens
                    print("cr_tokens", cr_tokens)
                    cr_list.append(cr_tokens)
    if enforce != 0:
        print("{} Records skipped due to enforce={} is = {}".format(fbase, enforce, skipCount))
    return y, cr_list


def PT_data():
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open('ubuntu_data/ubuntu_data/train.txt', 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    y = [int(a[0]) for a in data]
    cr = [[sen for sen in a[1:]] for a in data]
    crnew=[]
    for i,crsingle in enumerate(cr):
        if y[i]==1:
            crnew.append(crsingle)
    crnew=crnew
    pickle.dump(crnew, file=open("ubuntu_post_train.pkl", 'wb'))

if __name__ == '__main__':
    #Fine_tuning data constuction
    #including tokenization step
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased",do_lower_case=True)
    special_tokens_dict = {'additional_special_tokens': ['__eot__', '__eou__']}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)
    
    
    
    vocab = load_vocab('ubuntu_data/ubuntu_data/vocab.txt')
    response_dict = build_response_dict('ubuntu_data/ubuntu_data/responses.txt')

    train, test, dev = {}, {}, {}
    train['y'], train['cr'] = FT_data('ubuntu_data/ubuntu_data/train.txt', vocab, response_dict, tokenizer=bert_tokenizer)
    dev['y'], dev['cr'] = FT_data('ubuntu_data/ubuntu_data/valid.txt', vocab, response_dict, tokenizer=bert_tokenizer)
    test['y'], test['cr']= FT_data('ubuntu_data/ubuntu_data/test.txt',vocab, response_dict, tokenizer=bert_tokenizer)
    #char_vocab = defaultdict(float)
    dataset = train, dev, test
    pickle.dump(dataset, open('ubuntu_data/ubuntu_data/ubuntu_dataset_1M.pkl', 'wb'))


    #posttraining data construction
    #does not include tokenization step
    PT_data()