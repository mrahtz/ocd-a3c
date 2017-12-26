#!/bin/bash

for i in {1..15}; do
	python -u ./run.py 16 $i $* &> ${i}.log &
done
python -u ./run.py 16 0 $* &> 0.log
