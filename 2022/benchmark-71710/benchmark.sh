#!/bin/bash

python perf_lab_bcast.py --dtype_str double --op_str torch.eq
python perf_lab_bcast.py --dtype_str float --op_str torch.eq
python perf_lab_bcast.py --dtype_str half --op_str torch.eq
python perf_lab_bcast.py --dtype_str int --op_str torch.eq
python perf_lab_bcast.py --dtype_str uint8 --op_str torch.eq
python perf_lab_bcast.py --dtype_str double --op_str torch.add
python perf_lab_bcast.py --dtype_str float --op_str torch.add
python perf_lab_bcast.py --dtype_str half --op_str torch.add
python perf_lab_bcast.py --dtype_str int --op_str torch.add
python perf_lab_bcast.py --dtype_str uint8 --op_str torch.add
python perf_lab_bcast.py --dtype_str double --op_str torch.add --type_promotion True
python perf_lab_bcast.py --dtype_str float --op_str torch.add --type_promotion True
python perf_lab_bcast.py --dtype_str half --op_str torch.add --type_promotion True
python perf_lab_bcast.py --dtype_str int --op_str torch.add --type_promotion True
python perf_lab_bcast.py --dtype_str uint8 --op_str torch.add --type_promotion True
