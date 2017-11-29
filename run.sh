#!/bin/bash

for i in 1; do
	python -u ./run.py 2 $i &> ${i}.log &
done
python -u ./run.py 2 0 &> 0.log
