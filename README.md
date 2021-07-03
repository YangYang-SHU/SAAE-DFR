# SAAE-DFR
This is a code implemention for paper "Self-Attention Autoencoder for Anomaly Segmentation"

## Dataset
The MVTec AD dataset is available at:

https://www.mvtec.com/company/research/datasets/mvtec-ad

## Pretrained-ViT
We use pretrained ViT-B16 in the paper. You can get it at https://github.com/asyml/vision-transformer-pytorch. 



## Train
    python main.py --mode train --device cuda:0 --batch_size 32 --epochs 700
    
## Evaluation
    python main.py --mode evaluation

## Results
|     Class     |  ROC-AUC  |  PRO-AUC  |
|     :----:    |  :----:   |  :----:   |
|     Carpet    |    97.9   |    93.1   |
|      Grid     |    98.6   |    96.4   |
|     Leather   |    99.6   |    98.7   |
|      Tile     |    97.3   |    92.7   |
|      Wood     |    97.6   |    95.4   |
| Mean textures |    98.2   |    95.3   |
|     Bottle    |    97.9   |    94.3   |
|     Cable     |    96.8   |    89.0   |
|     Capsule   |    98.2   |    92.9   |
|     Hazelnut  |    98.5   |    96.6   |
|    Metal Nut  |    97.6   |    91.7   |
|      Pill     |    98.1   |    97.1   |
|     Screw     |    98.9   |    94.6   |
|   Toothbrush  |    98.7   |    93.1   |
|   Transistor  |    96.0   |    88.2   |
|     Zipper    |    96.9   |    90.3   |
| Mean textures |    97.8   |    92.8   |
|      Mean     |    97.9   |    93.6   |

## Qualitative Results
!(https://github.com/YangYang-SHU/SAAE-DFR/blob/main/figs/qualitative_results.png)

## Reference
https://github.com/YoungGod/DFR

https://github.com/lucidrains/vit-pytorch

https://github.com/asyml/vision-transformer-pytorch
