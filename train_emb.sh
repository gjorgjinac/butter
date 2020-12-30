#! /usr/bin/bash

export PYTHONHASHSEED=42

#g_python3 BILSTM_CRF_train.py
#echo $1 
if [ -z "$1" ]
	then 
		echo "no argument"
		~/venvs/foodis5/bin/python3 BILSTM_CRF_train.py
	else
		echo "argument $1"
		~/venvs/foodis5/bin/python3 BILSTM_CRF_train.py --fold $1

fi
#~/IJS/gpop_env/bin/python3 BILSTM_CRF_train.py --fold $1
