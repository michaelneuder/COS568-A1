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


Next up, comparing different compressions

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
| 0.25|  86.17  |  88.76    |  86.32      |  84.64    |  86.20        |
| 0.5|  84.60  |  88.14    |  85.09      |  84.58    |  86.35        |
| 1|  81.73  |  85.02    |  80.19      |  79.21    |  44.20        |

***Total inference time on testing dataset***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |
|-|-|-|-|-|-|
| 0.25|  0.2314  |  0.1940    |  0.1825      |  0.1862    |  0.2205        |
| 0.5|  0.2110  |  0.1963    |  0.1670      |  0.1797    |  0.2172        |
| 1|  0.2715  |  0.2252    |  0.1800      |  0.1982    |  0.1794        |


***FLOP***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |
|-|-|-|-|-|-|
| 0.25|  22,878,897  |  23,359,962    |  25,428,000      |  22,608,537    |  31,249,464        |
| 0.5|  12,774,993  |  13,466,235    |  15,884,286      |  14,202,177    |  25,214,127        |
| 1|  3,955,143  |  4,531,278    |  6,520,767      |  5,864,538    |  15,290,073        |


## Task 2: Tracing a single inference step

Cifar10, ResNet20, snip pruning
- lottery model
- compression 1 (sparsity 10^-1)


```
python main.py \
--model resnet20 \
--model-class lottery \
--dataset cifar10 \
--experiment singleshot \
--pruner snip \
--compression 1.0 \
--save-trace \
--expid save_trace_snip
```