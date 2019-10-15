import json
import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from given metadata.')
    parser.add_argument('--savedir', type=str, help='directory to save images')
    parser.add_argument('--metadata', type=str, metavar='*.csv', help='csv file which contains metadata(urls)')
    parser.add_argument('--tag_list', nargs='+', help='Tags that you want to parse')
    parser.add_argument('--ignore', nargs='*', default=[], help='Tags that you do not want to parse')
    args = parser.parse_args()

    tag_list = args.tag_list
    ignore = ['pokemon', 'chibi', 'monochrome'] + args.ignore
    checks = len(tag_list)
    data = pd.read_csv(args.metadata)

    parsed_ids = []
    for i in range(len(data['id'])):
        if data['rating'][i] != 's':
            continue
        tags = data['tags'][i].split(' ')
        tag_check = 0
        for tag in tags:
            if tag in ignore:
                break
            if tag in tag_list:
                tag_check += 1
            if tag_check == checks:
                parsed_ids.append(i)
                break

    parsed_data = data.iloc[parsed_ids]
    parsed_data.to_csv(os.path.join(args.savedir, 'parsed_data_{}.csv'.format('_'.join(tag_list))), index=False)
