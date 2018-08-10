import json
import sys
import os
import numpy as np
import scipy.sparse.linalg as splinalg
from ancillary import list_recursive

DEFAULT_k1 = 125
DEFAULT_k2 = 200
DEFAULT_data_file = 'data'


def local_noop(args):

    computation_output = dict(output=dict(computation_phase="local_noop"), )

    return json.dumps(computation_output)


def local_preprocess(args, data_file=DEFAULT_data_file, **kwargs):

    data = np.load(os.path.join(args["state"]["baseDirectory"], data_file))
    data = data[:, 0:500, :]

    data_mean = np.mean(data, 1)
    data_mean = np.reshape(data_mean, (data.shape[0], 1, data.shape[2]))

    data = data - np.tile(data_mean, (1, data.shape[1], 1))

    computation_output = dict(
        output=dict(computation_phase="local_preprocess"),
        cache=dict(all_data=data.tolist(), ), )

    return json.dumps(computation_output)


def local_subject_pca(args, k1=DEFAULT_k1, **kwargs):

    all_data = np.array(args['cache']['all_data'])
    red_data = np.zeros((all_data.shape[0], all_data.shape[1], k1))

    for subject in range(all_data.shape[0]):
        data = all_data[subject, :, :]
        w, v = splinalg.eigs(data.T.dot(data), k=k1)
        red_data[subject, :, :] = data @ v

    computation_output = dict(
        output=dict(computation_phase="local_subject_pca"),
        cache=dict(all_data=red_data.tolist(), ), )

    return json.dumps(computation_output)


def local_site_pca(args, k2=DEFAULT_k2, **kwargs):

    all_data = np.array(args['cache']['all_data'])
    shape = all_data.shape

    # Concatenate in the time dimension
    all_data = all_data.transpose(1, 0, 2).reshape(shape[1], -1)

    w, v = splinalg.eigs(all_data.T.dot(all_data), k=k2)
    red_data = all_data @ v
    red_data = np.real(red_data)

    computation_output = dict(
        output=dict(computation_phase="local_site_pca"),
        cache=dict(data=red_data.tolist(), ), )

    return json.dumps(computation_output)


def local_return_data(args):

    data = None
    if args['state']['clientId'] == args['input']['flag']:
        data = args['cache']['data']

    computation_output = dict(
        output=dict(
            computation_phase='local_return_data',
            data=data, ), )

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_noop(parsed_args)
    elif 'remote_determine_site_order' in phase_key:
        computation_output = local_preprocess(parsed_args)
        computation_output = local_subject_pca(json.loads(computation_output))
        computation_output = local_site_pca(json.loads(computation_output))
    elif 'remote_ping_sites' in phase_key:
        computation_output = local_return_data(parsed_args)
    else:
        raise ValueError('Phase error occurred at LOCAL')

    sys.stdout.write(computation_output)
