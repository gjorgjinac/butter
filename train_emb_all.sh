#! /usr/bin/bash

export PYTHONHASHSEED=42


~/venvs/foodis5/bin/python3 BILSTM_CRF_train.py --fold 1
~/venvs/foodis5/bin/python3 BILSTM_CRF_train.py --fold 2
~/venvs/foodis5/bin/python3 BILSTM_CRF_train.py --fold 3
~/venvs/foodis5/bin/python3 BILSTM_CRF_train.py --fold 4
~/venvs/foodis5//bin/python3 BILSTM_CRF_train.py --fold 5



