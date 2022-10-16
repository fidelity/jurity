#! /bin/bash

for alpha in 0.1 0.5 1 10 20 40 60 80; do
	for regularization in 0.02 0.05 0.1 0.2; do
		for factors in 50 64 100; do
			python submission.py --factors=$factors --regularization=$regularization --alpha=$alpha
		done
	done
done
