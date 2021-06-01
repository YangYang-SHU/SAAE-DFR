import argparse
from anoseg_dfr import AnoSegDFR
import os


def config():
    parser = argparse.ArgumentParser(description="Settings of DFR")

    # positional args
    parser.add_argument('--mode', type=str,choices=["train", "evaluation"],
                       default="train", help="train or evaluation")

    # general
    parser.add_argument('--model_name', type=str, default="", help="specifed model name")
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help="saving path")
    parser.add_argument('--img_size', type=int, nargs="+", default=(384, 384), help="image size (hxw)")
    parser.add_argument('--device', type=str, default="cuda:0", help="device for training and testing")
    parser.add_argument('--ViT_layers', type=int, default=4, help="ViT attention layers to use")

    parser.add_argument('--upsample', type=str, default="bilinear", help="operation for resizing cnn map")
    parser.add_argument('--is_agg', type=bool, default=False, help="if to aggregate the features")
    parser.add_argument('--featmap_size', type=int, nargs="+", default=(384, 384), help="feat map size (hxw)")
    parser.add_argument('--kernel_size', type=int, nargs="+", default=(4, 4), help="aggregation kernel (hxw)")
    parser.add_argument('--stride', type=int, nargs="+", default=(4, 4), help="stride of the kernel (hxw)")
    parser.add_argument('--dilation', type=int, default=1, help="dilation of the kernel")
    parser.add_argument('--smooth', type=str, default="", help="anomaly score smooth para, (_xxx)")


    # training and testing
    # default values
    data_name = "cable"
    train_data_path = "/home/anomaly_detection/datasets/mvtec/" + data_name + "/train"
    test_data_path = "/home/anomaly_detection/datasets/mvtec/" + data_name + "/test"

    parser.add_argument('--data_name', type=str, default=data_name, help="data name")
    parser.add_argument('--train_data_path', type=str, default=train_data_path, help="training data path")
    parser.add_argument('--test_data_path', type=str, default=test_data_path, help="testing data path")

    # CAE
    parser.add_argument('--latent_dim', type=int, default=None, help="latent dimension of CAE")
    parser.add_argument('--is_bn', type=bool, default=True, help="if using bn layer in CAE")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=700, help="epochs for training")    # default 700, for wine 150

    # segmentation evaluation
    parser.add_argument('--thred', type=float, default=0.5, help="threshold for segmentation")
    parser.add_argument('--except_fpr', type=float, default=0.005, help="fpr to estimate segmentation threshold")

    args = parser.parse_args()

    return args
    #python main.py --mode train --device cuda:0 --data_name grid --latent_dim
    #python main.py --mode evaluation --device cuda:1 --data_name zipper --latent_dim 610
if __name__ == "__main__":

    #########################################
    #    On the whole data
    #########################################
    cfg = config()
    #cfg.model_name = "/home/anomaly_detection/vit"

    # feature extractor
#     cfg.cnn_layers = ('relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')
    # dataset
    textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
    objects = ['bottle','cable', 'capsule','hazelnut', 'metal_nut',
                'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    data_names =  objects #textures
    
    dims = [113, 213, 105, 156, 168, 126, 165, 108, 199, 168, 287, 249, 278, 260, 267] # [287, 249, 278, 260, 267]
    # train or evaluation
    for data_name in data_names:
        i = data_names.index(data_name)
        cfg.latent_dim = dims[i]
        cfg.data_name = data_name
        cfg.train_data_path = "../datasets/mvtec/" + data_name + "/train"
        cfg.test_data_path = "..//datasets/mvtec/" + data_name + "/test"
        print(cfg.train_data_path)
        print(cfg.test_data_path)

        dfr = AnoSegDFR(cfg)
        if cfg.mode == "train":
            print('----train mode----------')
            dfr.train()
            dfr.metrics_evaluation()
            dfr.segmentation_results()
        else:
        #     print('----test mode----------')
            dfr.metrics_evaluation()
            #dfr.segmentation_evaluation()
            #dfr.segmentation_results()
            #dfr.segment_evaluation_with_fpr()
