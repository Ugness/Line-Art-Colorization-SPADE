import json
import os
import threading
from time import sleep
import pandas as pd
import argparse
import requests
from tqdm import tqdm

lock = threading.Semaphore(8)

def download(url, file):
    try:
        response = requests.get('http:' + data['sample_url'][i], stream=True, timeout=60)
    except requests.ReadTimeout:
        lock.release()
        return
    if response.status_code == 200:
        with open(os.path.join(savedir, '{}.png'.format(data['id'][i])), 'wb') as f:
            f.write(response.content)
    lock.release()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from given metadata.')
    parser.add_argument('--savedir', type=str, help='directory to save images')
    parser.add_argument('--metadata', type=str, metavar='*.csv', help='csv file which contains metadata(urls)')
    args = parser.parse_args()

    data = pd.read_csv(args.metadata)
    savedir = os.path.join(args.savedir, 'color')
    os.makedirs(savedir, exist_ok=True)

    # i = 0
    pbar = tqdm(total=len(data['sample_url']))
    threads = []
    for i in range(len(data['sample_url'])):
        url = 'http:' + data['sample_url'][i]
        file = os.path.join(savedir, '{}.png'.format(data['id'][i]))
        pbar.update(1)
        thread = threading.Thread(target=download, kwargs={'url': url, 'file': file})
        threads.append(thread)
        thread.start()

        lock.acquire()

    for t in threads:
        t.join()
    """
    while True:
        threads = []
        for j in range(64):
            if i > len(data['sample_url']):
                break
            url = 'http:' + data['sample_url'][i]
            file = os.path.join(savedir, '{}.png'.format(data['id'][i]))
            i += 1
            pbar.update(1)
            thread = threading.Thread(target=download, kwargs={'url': url, 'file': file})
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()
        if i > len(data['sample_url']):
            break
    """