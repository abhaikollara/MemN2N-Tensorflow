# MemN2N-Tensorflow
##Work under progress !

An Tensorflow implementation of End-to-End Memory Networks for language modelling. The original paper can be found a https://arxiv.org/abs/1503.08895v4.

The Penn Tree Bank is the used dataset.

Note : 
- The original paper uses a modified version of SGD as the optimizer, for simplicity an Adam optimizer with a learning rate of 0.001 has been used here.
- ReLU has been applied to the whole layer instead of half the layers as suggested in the paper
- Test functionality is still under construction
