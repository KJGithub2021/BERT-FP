# BERT-FP Updated Implementation on Ubuntu Corpus
This repository includes an implementation of Fine-grained Post-training for Improving Retrieval-based Dialogue Systems with faster training and evaluation. The latest version of Python (Python 3.12) may be used.

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
1. Setting up the TPU
```python
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```

2. Install Necessary Libraries
```python
!pip install setproctitle
!pip install transformers
!pip install accelerate
import os
os.environ.pop('TPU_PROCESS_ADDRESSES')
os.environ.pop('CLOUD_TPU_TASK_ID')
```

3. Configuring the Accelerate Library \
Using this library is optional but it will reduce the training and evaluation time of the model.
```python
!accelerate launch --tpu --num_processes 4 \
--num_machines 8 \
--main_training_function main \
 Response_selection.py
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

Be sure to save the checkpoints, ubuntu.0.pt, for future use.

## Acknowledgements
[BERT-FP Github Repository](https://github.com/hanjanghoon/BERT_FP) \
[Fine-grained Post-training for Improving Retrieval-based Dialogue Systems](https://aclanthology.org/2021.naacl-main.122/)
