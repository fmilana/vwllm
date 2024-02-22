from bs4 import BeautifulSoup
from collections import defaultdict
import os
import re
import time
import pandas as pd


def load_data(txt_path):
    start_time = time.time()
    print(f'Started at {time.ctime(start_time)}')
    csv_path = re.sub(r'\.txt$', '.csv', txt_path)
    # check if a csv exists
    if os.path.exists(csv_path):
        print('Loading from csv...')
        df = pd.read_csv(csv_path)
        return df.to_dict('list')
    
    print('No csv found, creating one...')
    # load the txt
    with open(txt_path, 'r', encoding='ISO-8859-1') as f:
        # total_chars = os.path.getsize(txt_path)
        # fraction = total_chars // 10
        # content = f.read(fraction)
        content = f.read()

    soup = BeautifulSoup(content, 'lxml-xml')
    
    data_defaultdict = defaultdict(list)
    for tag in soup.find_all():
        if not any(tag.name.lower().startswith(prefix) for prefix in ['body', 'text', 't', 'u', 'p']):
            text = ' '.join(tag.text.strip().split())
            data_defaultdict[tag.name].append(text)

    # pad shorter lists in data_defaultdict with None
    max_len = max(len(lst) for lst in data_defaultdict.values())
    for key in data_defaultdict:
        data_defaultdict[key] += [None] * (max_len - len(data_defaultdict[key]))

    data_dict = dict(data_defaultdict)

    df = pd.DataFrame(data_dict)

    print('Saving to csv...')
    df.to_csv(csv_path, index=False)

    end_time = time.time()
    print(f'Finished at {time.ctime(end_time)}. Time taken: {(end_time - start_time) / 60:.2f} minutes')
    
    return data_dict


load_data("data/Tagged ST VW 1927.txt")