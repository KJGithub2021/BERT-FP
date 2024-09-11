
-----NOT FINAL YET. ONLY ADDED IMPORTANT DETAILS----
# Download the following libraries
numpy
transformers
tqdm
torch

## Data Processing
```python
[Ubuntu Dialogue Corpus](https://drive.google.com/drive/folders/1cm1v3njWPxG5-XhEUpGH25TMncaPR7OM?usp=sharing) \
python Data_processing.py
[ubuntu_dataset_1M.pkl](https://drive.google.com/file/d/1wXU8-WdsWKqHY_wvJtdSCuju1RtBUB7y/view?usp=sharing)
[ubuntu_post_train.pkl](https://drive.google.com/file/d/1bhDVLQKQY_fViqFE7D8qiFDSN8vYU4GO/view?usp=sharing)

```

## Post-Training
Run python -u FPT/ubuntu_final.py --num_train_epochs 25

## Fine Tuning
Run python -u Fine-Tuning/Response_selection.py --is_training True

## Evaluation
Run python -u Fine-Tuning/Response_selection.py --is_training False

## Calculate Rouge Score
Run Fine-Tuning/rouger.py

_______________



## Checkpoint Files
Download the file here: [pt_data](https://drive.google.com/file/d/18eSZ9Kztj8F0wQ8BrZnPj4Eu7tQk95vR/view?usp=sharing) \
Place it in the following directory: \FPT\PT_checkpoint/ubuntu25/ \
The remaining checkpoint files are in the repository.

## Download Post Preprocessing Files (Optional) 
If you would like to skip preprocessing and are using the Ubuntu Corpus dataset, you can download the following files and place them in the ubuntu_data directory. \
[ubuntu_dataset_1M.pkl](https://drive.google.com/file/d/1KHx4EHZRcjLXcF18Pmsf3i5OsN7-jHNC/view?usp=sharing) \
[ubuntu_post_train.pkl](https://drive.google.com/file/d/1R5qE6XSkVIOUykXjaheyo5r3bOkF8LkQ/view?usp=sharing)

## Datasets 
If you wish to go through the preprocessing stage, you may download the dataset here: [Ubuntu Dialogue Corpus](https://drive.google.com/drive/folders/1cm1v3njWPxG5-XhEUpGH25TMncaPR7OM?usp=sharing) \
This repository will also work for the Douban and e-commerce datasets. If you would like to train the model on these, add the dataset to douban_data e_commerce_data directories, respectively. Please note that this may require minor modifications as this model has not been tested on the aforementioned datasets.

## Setting up the Environment
This model is compaible with python 3.10 and tensorflow 2.10.


2. Install Necessary Libraries
```python
!pip install transformers
!pip install accelerate
import os
os.environ.pop('TPU_PROCESS_ADDRESSES')
os.environ.pop('CLOUD_TPU_TASK_ID')
```

## Training
```python
start = time.time()
train_model(train, dev)
end = time.time()
print("use time: ", (end - start) / 60, " min")
```
## Evaluation
Before running the code below, two flags must be updated in Response_selection.py. `is_training` should be set to false and `is_test` should be true. This will skip the training code and use the saved model to obtain evaluation metrics.
```python
start = time.time()
test_model(test)
end = time.time()
print("use time: ", (end - start) / 60, " min")
```

If you'd like to skip training and evaluate the model, use the checkpoint files provided \
[BERT-FP Checkpoint Files](https://drive.google.com/file/d/1-3BgHYeXcMDxE06BGEYX6zdr5SGRaqez/view?usp=drive_link)

## Acknowledgements
[BERT-FP Github Repository](https://github.com/hanjanghoon/BERT_FP) \
[Fine-grained Post-training for Improving Retrieval-based Dialogue Systems](https://aclanthology.org/2021.naacl-main.122/)
