# # https://huggingface.co/docs/transformers/en/tasks/sequence_classification
# from transformers import AutoTokenizer, DataCollatorWithPadding
# from preprocess import load_data
# import evaluate
# import numpy as np


# def preprocess_function(tokenizer, data):
#     return tokenizer(data["text"], padding="max_length", truncation=True)


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)


# model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # load_data returns a dictionary!!
# training_data = load_data("data/Tagged ST VW 1927.txt")

# # check ==> https://huggingface.co/docs/datasets/en/about_map_batch
# # tokenized_text = training_data.map(preprocess_function, batched=True)

# tokenized_text = preprocess_function(tokenizer, training_data)

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# accuracy = evaluate.load("accuracy")
