# MemN2N-Language Modelling

An Tensorflow implementation of End-to-End Memory Networks for language modelling. The original paper can be found a https://arxiv.org/abs/1503.08895v4. Edit the configuration in main.py

Penn Tree Bank is used as the training dataset.

Note : 
- The original paper uses a modified version of SGD as the optimizer, for simplicity an Adam optimizer with a learning rate of 0.001 has been used here.
- ReLU has been applied to the whole layer instead of half the layers as suggested in the paper
