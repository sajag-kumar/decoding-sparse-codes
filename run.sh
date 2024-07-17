#!/bin/bash

decs=('bp' 'bposd' 'nbp' 'nbposd')
cirqs=('surface_5_5_cl' 'surface_7_7_cl')

for dec in "${decs[@]}"; do
    for cirq in "${cirqs[@]}"; do
        python3 simulate.py -dec "$dec" -c "$cirq"
    done
done