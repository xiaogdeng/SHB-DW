## Exploring the Inefficiency of Heavy Ball as Momentum Parameter Approaches 1

This repository is the official implementation of paper **Exploring the Inefficiency of Heavy Ball as Momentum Parameter Approaches 1**.

### Requirements

```setup
# GPU environment required
torch>=1.10.0
torchvision>=0.11.1
numpy>=1.19.5
```

### Dataset

The MNIST, and CIFAR-10 datasets can be downloaded automatically by `torchvision.datasets`.

### Example Usage

```train
python main.py --model 'Logistic' --dataset 'MNIST' --batch_size 256 --lr 0.01 \
               --optimizer 'SHB_DW' --alpha 0.999  --beta 0.999 --beta_min 0.1 \
               --epochs 60 --seed 42 --weight_decay 0.01 
```

### Usage

```
usage: main.py [-h]
    [--model  {'ResNet18', 'ResNet34', 'MLP', 'Logistic'}]
    [--dataset  {CIFAR10, MNIST}]
    [--lr  Learning rate]
    [--batch-size  Batch size] 
    [--epochs  Epoch]
    [--alpha  descending warmup parameter]
    [--beta  heavy ball parameter]
    [--beta_min  heavy ball parameter]
    [--seed Random  seed]
```

#### Note

* We provide a demo bash script file `bashrun.sh`

