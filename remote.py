"""
Remote file for globalPCA
"""

import os
import sys
import json
import numpy as np
from ancillary import list_recursive


DEFAULT_data_file = 'data.txt'
DEFAULT_k3 = 100


def remote_determine_site_order(args, **kwargs):
    site_order = list(range(len(args['input'])))
    site_flags = site_order[:]
    current_site = 0
    np.random.shuffle(site_order)
    computation_output = dict(
        output=dict(
            computation_phase="remote_determine_site_order"
        ),
        cache=dict(
            site_order=site_order,
            site_flags=site_flags,
            current_site=current_site,
        ),
        success=True
    )
    return json.dumps(computation_output)


def remote_ping_sites(args, **kwargs):
    current_site = args['cache']['current_site']
    flag = args['cache']['site_flags'][current_site]
    computation_output = dict(
        output=dict(
            computation_phase="remote_ping_sites",
            flag=flag
        ),
        success=True
    )

    return json.dumps(computation_output)


def remote_check_finished(args, **kwargs):
    current_site = args['cache']['current_site'] + 1
    phase_key = 'remote_check_finished_false'
    if current_site > len(args['input']):
        phase_key = 'remote_check_finished_true'
    computation_output = dict(
        output=dict(
            computation_phase=phase_key
        ),
        cache=dict(
            current_site=current_site
        ),
        sucess=True,
    )

    return json.dumps(computation_output)


def remote_pca_reduce(args, k3=DEFAULT_k3, **kwargs):
    current_site = args['cache']['current_site']
    received_data = args['input'][current_site]['data']
    cached_data = args['cache']['data']
    data = np.concatenate(cached_data, received_data)
    w, v = scipy.linalg.eigs(data.T.dot(data), k=k3)
    data = data.dot(v)
    computation_output = dict(
        output=dict(
            computation_phase="remote_pca_reduce"
        ),
        cache=dict(
            data=data
        ),
        success=True,
    )

    return json.dumps(computation_output)


def remote_normalize_top_columns(args, **kwargs):
    computation_output = dict(
        output=dict(
            computation_phase="remote_normalize_top_columns"
        ),
        success=True,
    )

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:  # FIRST PHASE
        computation_output = remote_determine_site_order(parsed_args)
    elif 'local_site_pca' in phase_key or 'remote_check_finished_false' in phase_key:
        computation_output = remote_ping_sites(parsed_args)  # First site ping
    elif 'local_return_data' in phase_key:
        computation_output = remote_pca_reduce(parsed_args)
        computation_output = remote_check_finished(parsed_args, json.loads(computation_output))
        if computation_output['output']['computation_phase'] == 'remote_check_finished_true':
            computation_output = remote_normalize_top_columns(parsed_args, json.loads(computation_output))
    else:
        raise ValueError('Oops')
    sys.stdout.write(computation_output)