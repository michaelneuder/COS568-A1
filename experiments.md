## Task 1: hyperparameter tuning


notes: 
- on Cifar + resnet, started with the lottery model, on which training loops seems to take about 20s each on the ionic machines
  - Note that there is no default model on resnet20 - this is useful to notice. 


Cifar10, ResNet20, RAND pruning
- lottery model
- compression 1 (sparsity 10^-1)
- post epoch 100

```
python main.py \
--model resnet20 \
--model-class lottery \
--dataset cifar10 \
--experiment singleshot \
--pruner rand \
--compression 1 \
--post-epochs 100 \
--expid cifar_rand_comp1_post100
```

final accuaracy: 81.73

same command as above, but now with the pruner changes and new expids. 

```
--pruner mag --pre-epochs 200
--expid cifar_mag_comp1_post100

--pruner snip
--expid cifar_snip_comp1_post100

--pruner grasp
--expid cifar_grasp_comp1_post100

--pruner synflow
--expid cifar_synflow_comp1_post100
```