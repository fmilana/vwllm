from lxml import etree
import re
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


ELEMENT_TAG = 'SC'

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
            bio_tag = f'B-{ELEMENT_TAG}' if i == 0 and inside else (f'I-{ELEMENT_TAG}' if inside else 'O')
            result_list.append({'paragraph_id': paragraph_id, 'word': word, 'bio_tag': bio_tag})
    # process the children of the current element
    for child in element:
        _process_element(child, paragraph_id, result_list, inside)
    # process the tail of the current element
    if element.tail:
        tail_words = element.tail.split()
        for word in tail_words:
            result_list.append({'paragraph_id': paragraph_id, 'word': word, 'bio_tag': 'O'})


def _tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['word'], truncation=True, padding='max_length', max_length=10)
    
    labels = []
    for i, label in enumerate(examples['bio_tag']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word. (this token is the first token of the word)
            elif word_idx != previous_word_idx:
                label_ids.append(label)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # whether we are inside a {ELEMENT_TAG} entity or not.
            else:
                label_ids.append(label if label != 0 else -100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs
            

print('Parsing XML...')
root = etree.fromstring(xml_content, parser=etree.XMLParser(recover=True))

result_list = []

print(f'Processing {ELEMENT_TAG} elements...')
_find_paragraphs_and_process(root, result_list)

df = pd.DataFrame(result_list)

df.to_csv(f'data/Tagged ST VW 1927_{ELEMENT_TAG}.csv', index=False)
print(f'File data/Tagged ST VW 1927_{ELEMENT_TAG}.csv saved')

# split the dataset into train and test sets using StratifiedGroupKFold
stratified_group_kfold = StratifiedGroupKFold(n_splits=5)

label_map = {'B-SC': 0, 'I-SC': 1, 'O': 2}
# convert the 'bio_tag' column to integer type
df['bio_tag'] = df['bio_tag'].map(label_map)

for train_index, test_index in stratified_group_kfold.split(df, df['bio_tag'], df['paragraph_id']):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    # break the loop after the first iteration
    break

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Tokenizing and encoding...')
tokenized_train_dataset = train_dataset.map(_tokenize_and_align_labels, batched=True)
tokenized_test_dataset = test_dataset.map(_tokenize_and_align_labels, batched=True)

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