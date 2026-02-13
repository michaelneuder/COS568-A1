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

MNIST, FC, default
- compression 1 (sparsity 10^-1)
- post epoch 100

flags
```
--model fc \
--model-class default \
--dataset mnist \
--experiment singleshot \
--pruner rand \
--compression 1 \
--post-epochs 100 \
--expid mnist_rand_comp1_post100
```


***Testing accuracy (top 1)***

|   Data  |   Arch |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|-|-|-|-|-|-|--|
|Cifar10 | ResNet20 |  81.73  |  85.02    |   80.19     |  79.21   |     44.20  |
|MNIST| FC | 95.96   |   97.68   |   96.66     |  96.39    |       11.35  |


Nexy up, comparing different compressions

Cifar10, ResNet20
- lottery model
- compression 0.25, 0.5 (already have results for 1)
- post epoch 100


my command
```
python /n/fs/mn3265cos568/COS568/torch_demo/COS568-A1/main.py \
--model resnet20 \
--model-class lottery \
--dataset cifar10 \
--experiment singleshot \
--pruner rand \
--compression 0.25 \
--post-epochs 100 \
--expid cifar_rand_post100_comp025
```

***Testing accuracy (top 1)***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|-|-|-|-|-|-|
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |

***Total inference time on testing dataset***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|-|-|-|-|-|-|
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |


***FLOP***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|-|-|-|-|-|-|
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |