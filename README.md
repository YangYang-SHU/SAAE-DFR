# SAAE-DFR
This is a code implemention for paper "Self-Attention Autoencoder for Anomaly Segmentation"

## Dataset
The MVTec AD dataset is available at:

https://www.mvtec.com/company/research/datasets/mvtec-ad

## Pretrained-ViT
You can get pretrained-ViT at https://github.com/asyml/vision-transformer-pytorch. 

We use ViT-B_16 in the paper.

## Train
    python main.py --mode train --device cuda:0 --batch_size 32 --epochs 700
    
## Evaluation
    python main.py --mode evaluation

## Reference
https://github.com/YoungGod/DFR

https://github.com/lucidrains/vit-pytorch

https://github.com/asyml/vision-transformer-pytorch
