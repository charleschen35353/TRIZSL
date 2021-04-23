from utils import *
import os, sys
import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--first", help="size of the first input layer", type=int)
parser.add_argument("-s", "--second", help="size of the second input layer", type=int)
parser.add_argument("-d", "--num_data", help="amount of data per class", type=int)
parser.add_argument("-e", "--epoch", help="epochs training per attribute", type=int, default = 20)
parser.add_argument("-o", "--output", help="output to be written", default = "result.txt")

args = parser.parse_args()
f = args.first
s = args.second
dpc = args.num_data
epochs = args.epoch
output_pth = args.output


tf.get_logger().setLevel('ERROR')

data_per_class = [1,2,5,10,20,40,80,150,400,100000]
sp1 = collections.defaultdict(list)
sp3 = collections.defaultdict(list)
up1 = collections.defaultdict(list)
up3 = collections.defaultdict(list)


(train_prec1, train_prec3), (test_prec1, test_prec3) = run_experiment(40, dpc, layer_sizes = (f,s), epochs = epochs )

print("[INFO] Config: {} data per class, f = {}, s = {}".format(dpc, f, s))
print("[INFO] Seen Precision@1: {:.3f}%".format(train_prec1*100))
print("[INFO] Seen Precision@3: {:.3f}%".format(train_prec3*100))
print("[INFO] Unseen Precision@1: {:.3f}%".format(test_prec1*100))
print("[INFO] Unseen Precision@3: {:.3f}%".format(test_prec3*100))
print("")
output = open(os.path.join(root_dir, output_pth), 'a')
output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(f,s,dpc,train_prec1*100, train_prec3*100, test_prec1*100, test_prec3*100))
output.close()
