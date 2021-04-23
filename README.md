# Zero Shot Learning via Few Shot Learning

## Description

This project aims to conduct experiments on inductive zero-shot learning(ZSL) model with few amount of data. 

We use ILSVRC2012 pretrained Res101 as feature extrator, and adopt attribute embedding for each attribute. 

Our method uses triplet network with 1 or no hidden layers to embedd attribute features into vectors of varialbe sizes. 

Comparing to the benchmark of classifying each attribute naively which gives ~53% harmonic mean, we reached near ~62% which is a ~9% improvement.

We also have better achievement on GZSL(general zero shot learning) in terms of harmonic mean on AWA2 over SGMA(Zhu et al., 2019) and similar performance(63%) against the generative method LsrGAN (Vyas et al., 2020). 

Please refer to our [paper](https://github.com/charleschen35353/TRIZSL/blob/master/Zero-shot%20Learning%20under%20Low%20Resource%20Data.pdf) for more details. 

And [presentation](https://youtu.be/EcT0f-5LiGg)

## Performance

Our model reach the following peak stats on AWA2 dataset with proposed split: [link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)
with only 25% data (~150 data per class).

seen precision@1: 84.342%

seen precision@3: 96.447%

unseen precision@1: 50.248%

unseen precision@3: 92.430%

Harmonic mean performance: 62.9%

## Instructions 

Please make sure you have tensorflow 2.4 installed, with tf-addons. 

To conduct an experiment, please run the following: 

$python3 train.py -f [first layer size] -s [secon layer size] -d [data per class] -o [outputfile]

You may also run the provided jupyter notebook and modify details for your own purpose.

