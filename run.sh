#!/bin/bash

for i in {1..7}; do
	python -u ./run.py 8 $i &> ${i}.log &
done
python -u ./run.py 8 0 &> 0.log
