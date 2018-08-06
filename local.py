import json
import sys
import os
import glob
import numpy as np
import scipy
from ancillary import list_recursive

DEFAULT_k1 = 120
DEFAULT_k2 = 500
DEFAULT_data_file = 'data.txt'


def local_noop(**kwargs):
    """
        # Description:
            local_preproc - preprocess data by centering

        # PREVIOUS PHASE:
            NA

        # INPUT:

        # OUTPUT:

        # NEXT PHASE:
            remote_init_env
    """
    computation_output = dict(
        output=dict(
            computation_phase="local_noop"
        ),
    )

    return json.dumps(computation_output)


def local_preprocess(args, data_file=DEFAULT_data_file, **kwargs):
    """
        # Description:
            local_preprocess - preprocess data by centering

        # PREVIOUS PHASE:
            NA

        # INPUT:

        # OUTPUT:

        # NEXT PHASE:
            remote_init_env
    """

    data = np.loadtxt(os.path.join(args["state"]["baseDirectory"], data_file))
    data = data - np.matlib.repmat(np.mean(data, 0), 1, data.size[1])

    computation_output = dict(
        output=dict(
            computation_phase="local_preprocess"
        ),
        cache=dict(
            all_data=data,
        ),
    )

    return json.dumps(computation_output)


def local_subject_pca(args, k1=DEFAULT_k1, **kwargs):
    """
        # Description:
            local_pca - perform subject-level PCA

        # PREVIOUS PHASE:
            local_preprocess

        # INPUT:

        # OUTPUT:

        # NEXT PHASE:
            remote_init_env
    """
    all_data = args['cache']['all_data']
    for row in range(all_data.shape[0]):
        data = all_data[row, :]
        w, v = scipy.sparse.linalg.eigs(data.dot(data.T), k=k1)
        all_data[row, :] = data.dot(v)
    computation_output = dict(
        output=dict(
            computation_phase="local_noop"
            ),
        cache=dict(
            all_data=all_data,
        ),
    )
    
    return json.dumps(computation_output)


def local_site_pca(args, k2=DEFAULT_k2, **kwargs):
    """
        # Description:
            local_pca

        # PREVIOUS PHASE:
            NA

        # INPUT:

        # OUTPUT:

        # NEXT PHASE:
            remote_init_env
    """
    all_data = args['cache']['all_data']
    # Concatenate in the time dimension
    w, v = scipy.sparse.linalg.eigs(data.T.dot(data), k=k2)
    data = all_data.dot(v)
    computation_output = dict(
        output=dict(
            computation_phase="local_noop"
        ),
        cache=dict(
            data=data
        ),
    )

    return json.dumps(computation_output)


def local_return_data(args, flags=[], **kwargs):
    """
        # Description:
            local_pca

        # PREVIOUS PHASE:
            NA

        # INPUT:

        # OUTPUT:

        # NEXT PHASE:
            remote_init_env
    """
    data = None
    if args['state']['siteName'] in flags:
        data = args['cache']['data']
    computation_output = dict(
        output=dict(
            computation_phase="local_noop",
            data=data,
        ),
    )

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))
    if not phase_key:  # FIRST PHASE
        computation_output = local_preprocess(parsed_args, **parsed_args['input'])
        computation_output = local_subject_pca(parsed_args, **json.loads(computation_output))
        computation_output = local_site_pca(parsed_args, **json.loads(computation_output))
    elif 'remote_ping_sites' in phase_key:
        computation_output = local_return_data(**parsed_args['input'])
    else:
        raise ValueError('Phase error occurred at LOCAL')
    sys.stdout.write(computation_output)