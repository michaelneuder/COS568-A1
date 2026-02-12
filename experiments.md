## Task 1: hyperparameter tuning

Cifar10, ResNet20, RAND pruning
- lottery model
- compression 1 (sparsity 10^-1)
- post epoch 100



```
python main.py --model-class lottery --model resnet20 --dataset cifar10 --experiment singleshot --pruner rand --compression 1 --post-epoch 100
```