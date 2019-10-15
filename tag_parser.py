import json
import os
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv(os.path.join('safebooru', 'all_data.csv'))
    tag_list = ['solo', 'white_background']
    ignore = []
    checks = len(tag_list)

    parsed_ids = []
    for i in range(len(data['id'])):
        if data['rating'][i] != 's':
            continue
        tags = data['tags'][i].split(' ')
        tag_check = 0
        for tag in tags:
            if tag in ignore:
                continue
            if tag in tag_list:
                tag_check += 1
            if tag_check == checks:
                parsed_ids.append(i)

    parsed_data = data.iloc[parsed_ids]
    exit()


