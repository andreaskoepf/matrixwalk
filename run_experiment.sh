#!/bin/bash
# experiment 5: testing bigger LSTM network on 4096 2d and 3d matrix with pure random walk

th main.lua -w world_4096_3d.txt -model lstm -m model_4096_3d_lstm512_b200_l2 -l 2 -n 512 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_2d.txt -model lstm -m model_4096_2d_lstm512_b200_l2 -l 2 -n 512 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_3d.txt -model lstm -m model_4096_3d_lstm256_b200_l2 -l 2 -n 256 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_2d.txt -model lstm -m model_4096_2d_lstm256_b200_l2 -l 2 -n 256 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_3d.txt -model lstm -m model_4096_3d_lstm128_b200_l2 -l 2 -n 128 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_2d.txt -model lstm -m model_4096_2d_lstm128_b200_l2 -l 2 -n 128 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_3d.txt -model lstm -m model_4096_3d_lstm64_b200_l2 -l 2 -n 64 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_2d.txt -model lstm -m model_4096_2d_lstm64_b200_l2 -l 2 -n 64 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_3d.txt -model lstm -m model_4096_3d_lstm32_b200_l2 -l 2 -n 32 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1
th main.lua -w world_4096_2d.txt -model lstm -m model_4096_2d_lstm32_b200_l2 -l 2 -n 32 -seqlen 50 -batchsize 200 -maxiter 10000 -dropout 0 -maxnoturnsteps 1

