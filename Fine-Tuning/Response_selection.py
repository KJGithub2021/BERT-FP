import time
import argparse
import pickle
import os
from BERT_finetuning import NeuralNetwork
from setproctitle import setproctitle
import torch

setproctitle('BERT_FP')



#Dataset path.
FT_data={
    'ubuntu': 'ubuntu_dataset_1M.pkl',
    'douban': 'douban_data/douban_dataset_1M.pkl',
    'e_commerce': 'e_commerce_data/e_commerce_dataset_1M.pkl'
}
print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=64,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adamw.")
parser.add_argument("--epochs",
                    default=10,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="Fine-Tuning/FT_checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="./Fine-Tuning/scorefile.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")

args, unknown = parser.parse_known_args()
args.save_path += args.task + '.' + "0.pt"
args.score_file_path = args.score_file_path
# load bert


print(args)
print("Task: ", args.task)


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    print("Training Now")
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    with open(FT_data[args.task], 'rb') as f:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')
        # Check the type and structure of `test`
        print(f"Type of test: {type(test)}")
        
        # If it's a dictionary, print the keys
        if isinstance(test, dict):
            print(f"Keys in test: {list(test.keys())}")
        
        # If it's a list or other structure, print some content
        else:
            print(f"First item in test: {test[0]}")
        for i in range(0, 30, 10):
            token_ids = train['cr'][i] 
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            sentence = tokenizer.decode(token_ids, skip_special_tokens=False)
            print(f"Original tokens: {token_ids}")
            print(f"Translated sentence: {sentence}")

            token_ids = dev['cr'][i] 
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            sentence = tokenizer.decode(token_ids, skip_special_tokens=False)
            print(f"Original tokens: {token_ids}")
            print(f"Translated sentence: {sentence}")

            token_ids = test['cr'][i] 
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            sentence = tokenizer.decode(token_ids, skip_special_tokens=False)
            print(f"Original tokens: {token_ids}")
            print(f"Translated sentence: {sentence}")
            
    print("Loading Data done")
    end = time.time()
    print("use time: ", (end - start) / 60, " min")
    print("Cuda Available true? ", torch.cuda.is_available())
    start = time.time()
    if args.is_training==True:
        train_model(train,dev)
        test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




