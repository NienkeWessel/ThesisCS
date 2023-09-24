from os import path
from torch import torch
from tqdm import tqdm
from glob import glob


def get_device():
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")  # NOTE: for now, we fix this to CPU to allow multiprocessing using cpu threads
    return device


def chunkify(input_file, config):
    # based on: https://www.blopig.com/blog/2016/08/processing-large-files-using-python/
    for filename in tqdm(glob(input_file, recursive=True), desc='Chunkify', mininterval=0.1, unit='files',
                         disable=not config.get('progress')):
        if not path.isfile(filename):
            continue
        file_end = path.getsize(filename)
        with open(filename, 'br') as f:
            chunk_end = f.tell()
            while True:
                chunk_start = chunk_end
                f.seek(config.get('chunk size'), 1)
                f.readline()
                chunk_end = f.tell()
                yield chunk_start, chunk_end - chunk_start, filename
                if chunk_end > file_end:
                    break
