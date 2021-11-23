CUDA_VISIBLE_DEVICES=0,1  python examples/main_unsupervised.py -b 128 -a resnet50 -d market1501 --iters 400 --w 0.5 --momentum 0.1 --eps 0.4 --num-instances 16  --logs-dir ./example_final/market_unsupervised


