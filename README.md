# Dual-Cluster-Contrastive

### Installation

```shell
cd Dual-Cluster-Contrast
python setup.py develop
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets Market-1501, and MSMT17 from (https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/data.zip), which is provided by existing public code ClusterContrast(https://github.com/alibaba/cluster-contrast).
Then unzip them under the directory like
```
examples/data/market1501/Market-1501-v15.09.15
exmpales/data/msmt17/MSMT17_V1
```

## Training on supervised

```
CUDA_VISIBLE_DEVICES=0  python examples/main.py -b 128 -a resnet50 -d market1501 --momentum 0.1 --w 0.25 --num-instances 16 --logs-dir ./examples/market1501_supervised
CUDA_VISIBLE_DEVICES=0  python examples/main.py -b 128 -a resnet50 -d msmt17 --momentum 0.1 --w 0.5 --num-instances 16 --logs-dir ./examples/msmt17_supervised
```

## Training on unsupervised
```
CUDA_VISIBLE_DEVICES=0,1  python examples/main_unsupervised.py -b 128 -a resnet50 -d market1501 --iters 400 --w 0.5 --momentum 0.1 --eps 0.4 --num-instances 16  --logs-dir ./example/market1501_unsupervised
```
In inspired by the Cluster-Contrast, the MSMT 17 is conducted with four gpus

```
CUDA_VISIBLE_DEVICES=0,1,2,3  python examples/main_unsupervised.py -b 256 -a resnet50 -d msmt17 --iters 400 --w 0.5 --momentum 0.1 --eps 0.6 --num-instances 16  --logs-dir ./example/msmt_unsupervised
```
The code is implemented based on the public code:https://github.com/alibaba/cluster-contrast 
