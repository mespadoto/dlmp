#!/bin/bash

python 00_get_data.py > 00.log 2>&1
python 01_exp1.py > 01.log 2>&1
python 02_exp2.py > 02.log 2>&1
python 03_exp3.py > 03.log 2>&1
python 04_pre_exp4.py > 04.log 2>&1
bash 05_exp4_ptsne.sh > 05.log 2>&1
python 06_exp4.py > 06.log 2>&1
python 07_exp5.py > 07.log 2>&1
python 08_exp6.py > 08.log 2>&1
