# https://huggingface.co/docs/transformers/en/training
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import BertConfig, AutoTokenizer, TrainingArguments, Trainer
from span_marker import SpanMarkerModel


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


# model_name = "compnet-renard/bert-base-cased-literary-NER"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = SpanMarkerModel.from_pretrained(model_name)

# dataset = load_dataset("data/Tagged ST VW 1927.csv")

# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# training_args = TrainingArguments(output_dir="checkpoints", evaluation_strategy="epoch")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )

# trainer.train()


# dataset = load_dataset("data/Tagged ST VW 1927.csv")

# labels = [label for example in dataset["train"]["labels"] for label in example]

# num_labels = len(set(labels))
