# GRAPES
Code to run the experiments of the GRAPES optimizer 

# Paper
### Learning in Deep Neural Networks Using a Biologically Inspired Optimizer

Giorgia Dellaferrera, Stanislaw Wozniak, Giacomo Indiveri, Angeliki Pantazi, and Evangelos Eleftheriou

arXiv: https://arxiv.org/abs/2104.11604

# Requirements
We run the experiments with the following:

Numpy framework: Python 3.9.5, Numpy 1.19.5, Keras 2.5.0

Tensorflow framework: Python 3.7.9, Tensorflow 1.15.0

Pytorch framework: Python 3.8.11, PyTorch 1.9.1.

# Numpy experiments  
The main experiments are run through `numpy_grapes_main.py`. 

For example, to run MNIST with BP and SGD:
```
python numpy_grapes_main.py --mnist --learn_type BP \
    --n_runs 10 --train_epochs 200 --eta 0.1 --dropout 0.9 \
    --update_type SGD --batch_size 64 --w_init he_uniform \
    --start_size 256 --n_hlayers 2 --act_hidden relu 
``` 

Substitute `--learn_type BP` with `--learn_type BPgrapes` to train with GRAPES on top of BP. 

Substitute `--learn_type BP` with `--learn_type FA` to train with FA. 

Substitute `--learn_type BP` with `--learn_type FAgrapes` to train with GRAPES on top of FA. 

Substitute `--learn_type BP` with `--learn_type DFA` to train with DFA. 

Substitute `--learn_type BP` with `--learn_type DFAgrapes` to train with GRAPES on top of DFA. 

# Tensorflow experiments  
The main experiments are run through `tensorflow_resnet9.py`. 

For example, to run CIFAR10 with BP and Adam:
```
python tensorflow_resnet9.py --cifar10 --optimizer adam \
    --num_epochs 250 --learning_rate 0.01  \
    --batchnorm
```
To train with GRAPES on top of BP, add `--grapes_fc` and `--grapes_conv` to the arguments.

# Pytorch experiments  
The experiment can be run with the default setting in the Jupyter Notebook.
