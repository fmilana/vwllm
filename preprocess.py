from stringprep import in_table_d2
from lxml import etree
import re
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


ELEMENT_TAG = 'IIM'

with open('data/Tagged ST VW 1927.xml', 'r', encoding='ISO-8859-1') as f:
    xml_content = f.read()


def _find_paragraphs_and_process(element, result_list):
    # if the element is a paragraph, process it
    if isinstance(element.tag, str) and re.match(r'P\d{3}', element.tag):
        _process_element(element, element.tag, result_list)
    # else, recursively search for paragraphs in the children of the element
    else:
        for child in element:
            _find_paragraphs_and_process(child, result_list)


def _process_element(element, paragraph_id, result_list, inside=False):
    # update paragraph_id if the element is a paragraph
    match = re.match(r'P(\d{3})', element.tag)
    if match:
        paragraph_id = int(match.group(1))
    # if the element is a SC, set inside to True
    if element.tag == ELEMENT_TAG:
        inside = True
    # process the text of the current element
    if element.text:
        words = element.text.split()
        for i, word in enumerate(words):
            iob_tag = f'B-{ELEMENT_TAG}' if i == 0 and inside else (f'I-{ELEMENT_TAG}' if inside else 'O')
            result_list.append({'paragraph_id': paragraph_id, 'word': word, 'iob_tag': iob_tag})
    # process the children of the current element
    for child in element:
        _process_element(child, paragraph_id, result_list, inside)
    # process the tail of the current element
    if element.tail:
        tail_words = element.tail.split()
        for word in tail_words:
            result_list.append({'paragraph_id': paragraph_id, 'word': word, 'iob_tag': 'O'})
            

def _tokenize_and_preserve_labels(batch, label_all_tokens=True):
    tokenized_inputs = tokenizer(batch['paragraph'], truncation=True, padding=True)

    label_ids_list = [] 
    for i, word_labels in enumerate(batch['word_labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_labels = word_labels.split(',')
        
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label = word_labels[word_idx] if word_idx < len(word_labels) else 'O'  # Default to 'O' or another default label
                label_id = label2id[label]
                # If label_all_tokens is True OR this token is the first token of a word, assign the actual label ID
                if label_all_tokens or word_idx != previous_word_idx:
                    label_ids.append(label_id)
                # Otherwise, if it's a subsequent token of a word and label_all_tokens is False, set its label to -100
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        label_ids_list.append(label_ids)  # Append the list of label IDs for this sentence to the overall list

    tokenized_inputs["labels"] = label_ids_list
    return tokenized_inputs


print('Parsing XML...')
root = etree.fromstring(xml_content, parser=etree.XMLParser(recover=True))

result_list = []

print(f'Processing {ELEMENT_TAG} elements...')
_find_paragraphs_and_process(root, result_list)

df = pd.DataFrame(result_list)

# split the dataset into train and test sets using StratifiedGroupKFold
stratified_group_kfold = StratifiedGroupKFold(n_splits=5)

for train_index, test_index in stratified_group_kfold.split(df, df['iob_tag'], df['paragraph_id']):
    train_df = df.iloc[train_index].copy()
    test_df = df.iloc[test_index].copy()
    # break the loop after the first iteration
    break


train_df['paragraph'] = train_df[['paragraph_id', 'word', 'iob_tag']].groupby(['paragraph_id'])['word'].transform(lambda x: ' '.join(x))
train_df['word_labels'] = train_df[['paragraph_id', 'word', 'iob_tag']].groupby(['paragraph_id'])['iob_tag'].transform(lambda x: ','.join(x))
train_df = train_df[['paragraph', 'word_labels']].drop_duplicates().reset_index(drop=True)

test_df['paragraph'] = test_df[['paragraph_id', 'word', 'iob_tag']].groupby(['paragraph_id'])['word'].transform(lambda x: ' '.join(x))
test_df['word_labels'] = test_df[['paragraph_id', 'word', 'iob_tag']].groupby(['paragraph_id'])['iob_tag'].transform(lambda x: ','.join(x))
test_df = test_df[['paragraph', 'word_labels']].drop_duplicates().reset_index(drop=True)

train_df.to_csv(f'data/train_{ELEMENT_TAG}.csv', index=False)
test_df.to_csv(f'data/test_{ELEMENT_TAG}.csv', index=False)

label2id = {f'B-{ELEMENT_TAG}': 0, f'I-{ELEMENT_TAG}': 1, 'O': 2}
id2label = {0: f'B-{ELEMENT_TAG}', 1: f'I-{ELEMENT_TAG}', 2: 'O'}

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Tokenizing and encoding...')
tokenized_train_dataset = train_dataset.map(_tokenize_and_preserve_labels, batched=True)
tokenized_test_dataset = test_dataset.map(_tokenize_and_preserve_labels, batched=True)

tokenized_train_dataset.save_to_disk(f'data/tokenized_train_dataset_{ELEMENT_TAG}.hf')
tokenized_test_dataset.save_to_disk(f'data/tokenized_test_dataset_{ELEMENT_TAG}.hf')
print(f'Files data/tokenized_train_dataset_{ELEMENT_TAG}.hf and data/tokenized_test_dataset_{ELEMENT_TAG}.hf saved')

print('Preprocessing done')


# Print out a few examples from the tokenized datasets
# print("=========================== Train dataset:")
# for i in range(100):
#     print(f"Example {i+1}:")
#     print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokenized_train_dataset[i]['input_ids'])}")
#     print(f"Labels: {tokenized_train_dataset[i]['labels']}")
#     print()

# print("=========================== Test dataset:")
# for i in range(100):
#     print(f"Example {i+1}:")
#     print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokenized_test_dataset[i]['input_ids'])}")
#     print(f"Labels: {tokenized_test_dataset[i]['labels']}")
#     print()