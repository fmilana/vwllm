from bs4 import BeautifulSoup
import os
import re
import time
import pandas as pd


# still produces duplicates
def _process_tag(tag, data_dict):
    tag_name = 'P' if tag.name.startswith('P') else tag.name

    if not re.match(r'(TEI|text|T\d+|U\d+)', tag_name):
        if tag_name not in data_dict:
            data_dict[tag_name] = []

        text = ' '.join(tag.text.strip().split())
        data_dict[tag_name].append(text)

    children = tag.find_all(recursive=False)
    for child in children:
        _process_tag(child, data_dict)
        

def load_data(xml_path):
    start_time = time.time()
    print(f'Started at {time.ctime(start_time)}')
    csv_path = re.sub(r'\.xml$', '.csv', xml_path)
    print
    # check if a csv exists
    # if os.path.exists(csv_path):
    #     print('Loading from csv...')
    #     df = pd.read_csv(csv_path)
    #     return df.to_dict('list')
    
    print('No csv found, creating one...')
    # load the txt
    with open(xml_path, 'r', encoding='ISO-8859-1') as f:
        # total_chars = os.path.getsize(txt_path)
        # fraction = total_chars // 10
        # content = f.read(fraction)
        content = f.read()

    soup = BeautifulSoup(content, 'xml')

    data_dict = {}
    print(f'len(soup.find_all()): {len(soup.find_all())}')
    for tag in soup.find_all():
        _process_tag(tag, data_dict)

    print(f'len(data_dict): {len(data_dict)}')

    # pad shorter lists in data_defaultdict with None
    max_len = max(len(lst) for lst in data_dict.values())
    for key in data_dict:
        data_dict[key] += [None] * (max_len - len(data_dict[key]))

    df = pd.DataFrame(data_dict)

    print('Saving to csv...')
    df.to_csv(csv_path, index=False)

    end_time = time.time()
    print(f'Finished at {time.ctime(end_time)}. Time taken: {(end_time - start_time) / 60:.2f} minutes')
    
    return data_dict


load_data("data/Tagged ST VW 1927.xml")