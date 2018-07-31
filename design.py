"""

Local NoOp  ->  Remote Determine Site Order
Remote Determine Site Order  ->  Local Prepoc/Local Subject PCA/Local Site PCA
Local Preproc/Local Subject PCA/Local Site PCA -> Remote Ping Sites (1st in order)
Remote Ping Sites (1st in order) -> Local Return Data (1st site)
Local Return Data (1st site) -> Remote Check Finished (false)/Remote ping Sites (ith in order)
for site i = 2:num_sites:
    Remote Check Finished (false)/Remote ping Sites (ith in order) -> Local Return Data (ith site)
    Local Return Data (ith site) -> Remote reduce
    Remote reduce/Remote Check Finished (true/false) ->
        (true) remote normalize
        (false) local return data [line 9]

RESULT: V on aggregator node
"""
