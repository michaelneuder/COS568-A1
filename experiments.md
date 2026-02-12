## Task 1: hyperparameter tuning

Cifar10, ResNet20, RAND pruning
- lottery model
- compression 1 (sparsity 10^-1)
- post epoch 100

notes: 
- on Cifar + resnet, started with the lottery model, on which training loops seems to take about 20s each on the ionic machines
  - Also testing with the default, model, which might be a lot slower.


```
python main.py \
--model resnet20 \
--model-class default \
--dataset cifar10 \
--experiment singleshot \
--pruner rand \
--compression 1 \
--post-epoch 100 \
--expid cifar_rand_comp1_post100
```