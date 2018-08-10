"""
Remote file for globalPCA
"""
import collections
import sys
import json
import numpy as np
import scipy.sparse.linalg as splinalg
from ancillary import list_recursive

DEFAULT_data_file = 'data'
DEFAULT_k3 = 100


def remote_determine_site_order(args):

    site_order = list(args['input'].keys())
    np.random.shuffle(site_order)

    computation_output = dict(
        output=dict(computation_phase="remote_determine_site_order"),
        cache=dict(site_order=site_order, ), )
    return json.dumps(computation_output)


def remote_ping_sites(args, **kwargs):

    site_order = args["cache"]["site_order"]
    site_order = collections.deque(site_order)
    flag = site_order.popleft()

    if not site_order:
        conclude = 1
    else:
        conclude = 0

    computation_output = dict(
        output=dict(
            computation_phase="remote_ping_sites",
            flag=flag, ),
        cache=dict(site_order=list(site_order), flag=flag, conclude=conclude),
    )

    return json.dumps(computation_output)


def remote_pca_reduce(args, k3=DEFAULT_k3, **kwargs):

    conclude = args['cache']['conclude']
    flag = args['cache']['flag']
    received_data = np.array(args['input'][flag]['data'])

    try:
        cached_data = np.array(args['cache']['data'])
    except KeyError:
        cached_data = []

    try:
        data = np.concatenate((cached_data, received_data), axis=1)
    except ValueError:
        data = received_data

    w, v = splinalg.eigs(data.T.dot(data), k=k3)
    data = data.dot(v)
    data = np.real(data)

    computation_output = dict(
        output=dict(computation_phase="remote_pca_reduce", conclude=conclude),
        cache=dict(data=data.tolist()), )

    return json.dumps(computation_output)


def remote_normalize_top_columns(prev_output):

    data = np.array(prev_output['cache']['data'])
    data = data / np.linalg.norm(data, axis=1).reshape((-1, 1))

    computation_output = dict(
        output=dict(
            data=data.tolist(),
            computation_phase="remote_normalize_top_columns"),
        success=True, )

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:
        computation_output = remote_determine_site_order(parsed_args)
    elif 'local_site_pca' in phase_key:
        computation_output = remote_ping_sites(parsed_args)
    elif 'local_return_data' in phase_key:
        computation_output = remote_pca_reduce(parsed_args)
        loaded_output = json.loads(computation_output)
        if loaded_output['output']['conclude']:
            computation_output = remote_normalize_top_columns(loaded_output)
        else:
            computation_output = remote_ping_sites(parsed_args)

    else:
        raise ValueError('Oops')

    sys.stdout.write(computation_output)
