r"""
.. code-block:: none

    Usage:
        password_pipeline [options]

    Examples:
        passwords_pipeline -i inputfile.tmp -o outputfile.dict

    Standard Options:
        -i --input <path to file>       Specify the input file to be cleaned, or provide a glob pattern
        -o --output <path to file>      Specify the output file name.

"""

from torch import torch
import torch.nn.functional as functional
from LSTM import BiRnn
from build_datasets import CharacterDataset
from utils import get_device, chunkify
from global_variables import *
from multiprocessing import cpu_count, current_process, Pool
from tqdm import tqdm
from os import path, mkdir, walk
from shutil import rmtree
from inspect import cleandoc
from docopt import docopt
from hashlib import md5

version = '0.0.1 - LSTM model 15000 words 8 epochs'


def pytorch_predict(model, test_loader, device):
    model.eval()

    all_outputs = torch.tensor([], device=device)

    # deactivate autograd engine to reduce memory usage and increase speed
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            outputs = model(*inputs)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    # print(all_outputs)
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = functional.softmax(all_outputs, dim=1).cpu().numpy()

    return y_pred, y_pred_prob


def process_chunk_get_probabilities(filename, chunk_start, chunk_size, config):
    """
    Main processing loop - this will output per line the probability outputs of the LSTM model.
    Example:
    0.927849531173706 0.07215043157339096 123456789
    0.906269907951355 0.09373012185096741 qwertyuiop
    ...
    0.6650089621543884 0.33499106764793396 password1
    0.5999095439910889 0.40009042620658875 1q2w3e4r5t

    Parameters
    ----------
    filename
    chunk_start
    chunk_size
    config

    Returns
    -------
    Password classification probability output of the form:
        probability of being a password (positive match)
        probability of note being a password (negative match)
        'password'
    """

    results = []

    temp_folder = 'pipeline_tmp'
    temp_file = md5(filename.encode()).hexdigest()

    pid = current_process().pid

    device = get_device()
    model = BiRnn(lstm_input_size, hidden_state_size, batch_size=batch_size, output_dim=output_dim,
                  num_layers=num_sequence_layers, rnn_type=rnn_type)
    model.load_state_dict(torch.load("LSTM_trained_15000words_8epochs"))
    model = model.to(device)

    if config.get('verbose'):
        print(f'Clean_up ({pid}): starting {filename}, {chunk_start}, {chunk_size}')

    with open(filename, 'rb') as f:
        if config.get('verbose'):
            print(f'Clean_up ({pid}): seeking {filename}, {chunk_start}, {chunk_size}')
        f.seek(chunk_start)
        if config.get('verbose'):
            print(f'Clean_up ({pid}): splitting {filename}, {chunk_start}, {chunk_size}')
        password_candidates = f.read(chunk_size).splitlines()
    if config.get('verbose'):
        print(f'Clean_up ({pid}): processing {filename}, {chunk_start}, {chunk_size}')

    x_axis_passwords = password_candidates
    y = torch.zeros(len(x_axis_passwords), 2)
    data = CharacterDataset(x_axis_passwords, y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
    y_pred, y_pred_prob = pytorch_predict(model, data_loader, device)

    result = list(filter(lambda x: len(x[0]) > 8, zip(x_axis_passwords, y_pred_prob)))
    sorted_result = sorted(result, key=lambda x: x[1][0], reverse=True)

    if config.get('verbose'):
        print(f'Clean_up ({pid}): stopping {filename}, {chunk_start}, {chunk_size}')
    # Processed all lines, flush everything
    with open(path.join(temp_folder, f'{temp_file}_{chunk_start}_result.txt'), 'a') as f:
        for item in sorted_result:
            password = item[0]
            password_probability = item[1][0]
            non_password_probability = item[1][1]
            f.write(f"{password_probability} {non_password_probability} {password.decode('utf8')}\n")
    if config.get('verbose'):
        print(f'Clean_up ({pid}): done {filename}, {chunk_start}, {chunk_size}')


def main():
    arguments = docopt(cleandoc('\n'.join(__doc__.split('\n')[2:])))

    if arguments.get('--input') and arguments.get('--output'):
        input_file = arguments.get('--input')
        output_file = arguments.get('--output')
    else:
        print(cleandoc('\n'.join(__doc__.split('\n')[2:])))
        exit()

    config = {
        'progress': True,
        'chunk size': 1024 * 1024,
        'use all threads': True,
    }

    print(f'Main: running pipeline - {version}')
    if path.isdir('pipeline_tmp'):
        rmtree('pipeline_tmp')
    mkdir('pipeline_tmp')

    if config.get('use all threads') is True:
        thread_count = cpu_count()
    else:
        thread_count = 1

    pool = Pool(thread_count)

    jobs = []

    print(f'Main: start chunking file {input_file}')
    for chunk_start, chunk_size, filename in chunkify(input_file, config):
        jobs.append(pool.apply_async(process_chunk_get_probabilities, (filename, chunk_start, chunk_size, config)))
    print('Main: done chunking file.')

    print(f'Main: start processing, running at a maximum {thread_count} thread(s).')
    for job in tqdm(jobs, desc='Main', mininterval=1, unit='chunks', disable=not config.get('progress')):
        job.get()

    pool.close()
    print('Main: done processing.')

    print('Main: start combining results.')
    p_output_file = open(output_file, 'w')

    for root, directories, files in walk('pipeline_tmp'):
        for file_name in files:
            if '_result.txt' in file_name:
                with open(path.join(root, file_name), 'r') as f:
                    p_output_file.write(f.read())

    p_output_file.close()
    print(f'Main: done combining results. Output found in {output_file}')

    rmtree('pipeline_tmp')

    return


main()
