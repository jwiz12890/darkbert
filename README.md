---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- conll2003
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: dark-bert-finetuned-ner
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: conll2003
      type: conll2003
      config: conll2003
      split: train
      args: conll2003
    metrics:
    - name: Precision
      type: precision
      value: 0.928300642821823
    - name: Recall
      type: recall
      value: 0.9478290138000673
    - name: F1
      type: f1
      value: 0.9379631942709634
    - name: Accuracy
      type: accuracy
      value: 0.9859009831047272
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# dark-bert-finetuned-ner

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the conll2003 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0639
- Precision: 0.9283
- Recall: 0.9478
- F1: 0.9380
- Accuracy: 0.9859

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0881        | 1.0   | 1756 | 0.0716          | 0.9172    | 0.9322 | 0.9246 | 0.9817   |
| 0.0375        | 2.0   | 3512 | 0.0610          | 0.9275    | 0.9455 | 0.9364 | 0.9857   |
| 0.0207        | 3.0   | 5268 | 0.0639          | 0.9283    | 0.9478 | 0.9380 | 0.9859   |


### Framework versions

- Transformers 4.22.1
- Pytorch 1.10.0
- Datasets 2.5.1
- Tokenizers 0.12.1
