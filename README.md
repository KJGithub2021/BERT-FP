# BERT-FP for Dialogue Systems

This repository contains the implementation of BERT-FP using Ubuntu Dialogue Corpus V2.

## Prerequisites
This model is compaible with python 3.12 and torch 2.6.0.dev20240923+cu121

Before you begin, ensure you have the following libraries installed:

```
numpy
transformers
tqdm
torch
evaluate
rouge
```

You can install these using pip:

```python
pip install numpy transformers tqdm evaluate rouge
```

## Data Processing

1. Download the [Ubuntu Dialogue Corpus](https://drive.google.com/drive/folders/1cm1v3njWPxG5-XhEUpGH25TMncaPR7OM?usp=sharing). Place the files in the ubuntu_data folder.

2. Process the data by running:

   ```python
   python Data_processing.py
   ```

3. After processing, you should have the following files:
   - [ubuntu_dataset_1M.pkl](https://drive.google.com/file/d/1wXU8-WdsWKqHY_wvJtdSCuju1RtBUB7y/view?usp=sharing)
   - [ubuntu_post_train.pkl](https://drive.google.com/file/d/1bhDVLQKQY_fViqFE7D8qiFDSN8vYU4GO/view?usp=sharing)

## Training Pipeline

### Post-Training

Run the following command:

```python
python -u FPT/ubuntu_final.py --num_train_epochs 25
```
You may skip this step by placing [bert.pt](https://drive.google.com/file/d/1XM1oRiwMnqW8-P-IS_heUBJykqW7VEdd/view?usp=sharing) in the FPT/PT_checkpoint folder.

### Fine-Tuning

To start fine-tuning, use:

```python
python -u Fine-Tuning/Response_selection.py --is_training True
```
You may skip this step by placing [ubuntu.0.pt]() in the Fine-Tuning/FT-checkpoint folder

## Evaluation

For evaluation, run:

```python
python -u Fine-Tuning/Response_selection.py --is_training False
```
This will print evaluation metrics and will generate a prediction-score file like [scorefile.txt]().

## Calculate ROUGE Score

To calculate the ROUGE score, execute:

```python
python compute_rouge.py
```

## Acknowledgements
[BERT-FP Github Repository](https://github.com/hanjanghoon/BERT_FP) \
[Fine-grained Post-training for Improving Retrieval-based Dialogue Systems](https://aclanthology.org/2021.naacl-main.122/)
