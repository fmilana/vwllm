# https://huggingface.co/docs/transformers/en/training
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification 


ELEMENT_TAG = 'SC'

model_name = 'google-bert/bert-base-uncased'

train_dataset_path = f'data/tokenized_train_dataset_{ELEMENT_TAG}.hf/data-00000-of-00001.arrow'
test_dataset_path = f'data/tokenized_test_dataset_{ELEMENT_TAG}.hf/data-00000-of-00001.arrow'

dataset = load_dataset('arrow', data_files={'train': train_dataset_path, 'test': test_dataset_path})

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir='checkpoints', 
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

trainer.train()

results = trainer.evaluate()
print(results)