#!/bin/bash

for i in {1..3}; do
	python -u ./run.py 4 $i &> ${i}.log &
done
python -u ./run.py 4 0 &> 0.log
