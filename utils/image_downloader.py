import json
import os
import threading
from time import sleep
import pandas as pd
import argparse
import requests
from tqdm import tqdm

lock = threading.Semaphore(200)


def download(url, file):
    tries = 0
    while tries < 5:
        try:
            response = requests.get(url, stream=True, timeout=3)
            tries += 1
            if response.status_code == 200:
                with open(file, 'wb') as f:
                    f.write(response.content)
            lock.release()
            return
        except:
            if tries >= 5:
                print("Failed to Download Image {}".format(file), flush=True)
                lock.release()
                return
            # print("Retry to Download Image {}".format(file), flush=True)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from given metadata.')
    parser.add_argument('--savedir', type=str, help='directory to save images')
    parser.add_argument('--metadata', type=str, metavar='*.csv', help='csv file which contains metadata(urls)')
    args = parser.parse_args()

    print("Loading csv")
    data = pd.read_csv(args.metadata)
    print("Done", flush=True)
    savedir = os.path.join(args.savedir, 'color')
    os.makedirs(savedir, exist_ok=True)

    # i = 0
    pbar = tqdm(total=len(data['sample_url']))
    threads = []
    for i in range(len(data['sample_url'])):
        url = 'http:' + data['sample_url'][i].lstrip('http:')
        file = os.path.join(savedir, '{}.png'.format(data['id'][i]))
        pbar.update(1)
        thread = threading.Thread(target=download, kwargs={'url': url, 'file': file})
        threads.append(thread)
        thread.start()
        lock.acquire()
        # sleep(0.5)

    for t in threads:
        t.join()
