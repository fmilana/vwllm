# https://huggingface.co/docs/transformers/en/training
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


ELEMENT_TAG = 'IIM'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model_name = 'google-bert/bert-base-uncased'

train_dataset_path = f'data/tokenized_train_dataset_{ELEMENT_TAG}.hf/data-00000-of-00001.arrow'
test_dataset_path = f'data/tokenized_test_dataset_{ELEMENT_TAG}.hf/data-00000-of-00001.arrow'

dataset = load_dataset('arrow', data_files={'train': train_dataset_path, 'test': test_dataset_path})

label2id = {f'B-{ELEMENT_TAG}': 0, f'I-{ELEMENT_TAG}': 1, 'O': 2}
id2label = {0: f'B-{ELEMENT_TAG}', 1: f'I-{ELEMENT_TAG}', 2: 'O'}


# def compute_metrics(p):
#     predictions = np.argmax(p.predictions, axis=-1)
#     true_labels = p.label_ids

#     # Convert indices to labels, ignoring -100 labels
#     pred_labels = [[id2label[p] for p, l in zip(pred, lbl) if l != -100] for pred, lbl in zip(predictions, true_labels)]
#     true_labels = [[id2label[l] for p, l in zip(pred, lbl) if l != -100] for pred, lbl in zip(predictions, true_labels)]

#     # Calculate metrics per tag
#     report = classification_report(true_labels, pred_labels)

#     return report

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(id2label), id2label=id2label, label2id=label2id)

# sanity check (initial_loss should close to ln(3) = 1.0986122886681098)
# ids = torch.tensor(dataset['train'][0]['input_ids']).unsqueeze(0)
# mask = torch.tensor(dataset['train'][0]['attention_mask']).unsqueeze(0)
# labels = torch.tensor(dataset['train'][0]['labels']).unsqueeze(0)
# ids = ids.to('cuda')
# mask = mask.to('cuda')
# labels = labels.to('cuda')
# outputs = model(ids, mask, labels=labels)
# initial_loss = outputs.loss
# print(f'initial_loss = {initial_loss}')
# # sanity check (tr_logits.shape should be (1, 512, 3))
# tr_logits = outputs.logits
# print(f'tr_logits.shape = {tr_logits.shape}')

training_args = TrainingArguments(
    output_dir=f'checkpoints/{ELEMENT_TAG}', 
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=1e-05,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    # compute_metrics=compute_metrics
)

trainer.train()

report = trainer.evaluate()

print(report)


# # to-do:
# omit [pad] tokens 


# # make sure not to overfit, use train test validate

# # Class weights: You can assign higher weights to the minority class during training. In the Hugging Face Trainer, you can do this by setting the compute_loss function to use class weights.