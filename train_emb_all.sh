#! /usr/bin/bash

export PYTHONHASHSEED=42


~/IJS/gpop_env/bin/python3 BILSTM_CRF_train.py --fold 1
~/IJS/gpop_env/bin/python3 BILSTM_CRF_train.py --fold 2
~/IJS/gpop_env/bin/python3 BILSTM_CRF_train.py --fold 3
~/IJS/gpop_env/bin/python3 BILSTM_CRF_train.py --fold 4
~/IJS/gpop_env/bin/python3 BILSTM_CRF_train.py --fold 5



