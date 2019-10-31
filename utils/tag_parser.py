import json
import os
import pandas as pd
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from given metadata.')
    parser.add_argument('--savedir', required=True, type=str, help='directory to save parsed_metadata')
    parser.add_argument('--metadata', required=True, type=str, metavar='*.csv', help='csv file which contains metadata(urls)')
    parser.add_argument('--tag_list', nargs='+', help='Tags that you want to parse')
    parser.add_argument('--ignore', nargs='*', default=[], help='Tags that you do not want to parse')
    args = parser.parse_args()

    tag_list = args.tag_list
    ignore = ['pokemon', 'chibi', 'monochrome'] + args.ignore
    checks = len(tag_list)
    print("Loading csv")
    data = pd.read_csv(args.metadata)
    print("Done", flush=True)

    parsed_ids = []
    for i in tqdm(range(len(data['id']))):
        if data['rating'][i] != 's':
            continue
        tags = data['tags'][i].split(' ')
        tag_check = 0
        ign = False
        for tag in tags:
            if tag in ignore:
                ign = True
                break
        if not ign:
            for tag in tags:
                if tag in tag_list:
                    tag_check += 1
                if tag_check == checks:
                    parsed_ids.append(i)
                    break

    parsed_data = data.iloc[parsed_ids]
    print("# of Collected Images : {}".format(len(parsed_data['id'])))
    parsed_data.to_csv(os.path.join(args.savedir, 'parsed_data_{}.csv'.format('_'.join(tag_list))), index=False)
    print("File saved at {}".format(os.path.join(args.savedir, 'parsed_data_{}.csv'.format('_'.join(tag_list)))))
