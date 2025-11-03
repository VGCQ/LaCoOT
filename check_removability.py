import os 
import argparse
import torch
import torchvision
import random
import numpy as np
import scipy
import pandas as pd 
from ptflops import get_model_complexity_info


from models.ResNet import ResNet, BasicBlock, ResNet_orig
from models.Mobilenetv2 import MobileNetV2
from train import val_one_epoch
from dataloaders.cifar10 import get_cifar10
from dataloaders.tinyimagenet200 import get_tinyimagenet200

def main():
    parser = argparse.ArgumentParser(description='Check Layer Removability')
    parser.add_argument('--name_checkpoint', default='checkpoints/checkpoint_CIFAR-10_ResNet-18_lambda_5.0_last_epoch.pt',  help='path to checkpoint (default: checkpoints/checkpoint_CIFAR-10_ResNet-18_lambda_5.0_last_epoch.pt)')
    parser.add_argument('--model', default='ResNet-18',  help='model (default: ResNet-18)')
    parser.add_argument('--dataset', default='CIFAR-10',  help='dataset (default : CIFAR-10)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='seed (default: 0)')
    parser.add_argument('--optimizer', default="SGD",  help='Optimizer Adam/SGD (default: SGD)')
    parser.add_argument('--momentum', type=float, default=0.9,  help='Momentum (default: 0.9)')
    parser.add_argument('--epochs', type=int, default=160,  help='Epochs (default: 160)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size (default:128)')
    parser.add_argument('--lr', type=float, default=0.1,  help='Learning Rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.1,  help='Drop factor (default: 0.1)')
    parser.add_argument('--milestones', default="80,120",  type=lambda s: [int(item) for item in s.split(',')], help='Milestones (default: "80,120")')
    parser.add_argument('--wd', type=float, default=1e-4,  help='Weight decay (default: 1e-4)')
    parser.add_argument('--dir_to_save_checkpoint',type=str, default='checkpoints/', help="directory to save checkpoint (default:'checkpoints/')")
    parser.add_argument('--device', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('--root', default="data/", help='root of dataset (default:"data/")')
    parser.add_argument('--distributed', default=False, help='distributed (default : False)')
    parser.add_argument('--workers', type=int, default=4, help='workers (default:4)')

    parser.add_argument('--lambda_reg', type=float, default=1, help='Lambda (default:1)')
    
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--cache-dataset", dest="cache_dataset", 
                        help="Cache the datasets for quicker initialization. It also serializes the transforms",
                        action="store_true")
    parser.add_argument("--test-only",dest="test_only",help="Only test the model",action="store_true")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    
    args = parser.parse_args()

    ## Getting the desired GPU
    cuda = "cuda:"+str(args.device)
    device = torch.device(cuda)

    ## SEEDING
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    
    ## DATASET:
    if args.dataset == "CIFAR-10":
        train_loader, val_loader, test_loader, num_classes, _ = get_cifar10(args, size_train_set = 0.9, get_train_sampler = False, transform_train = True)
        size = (3, 32, 32)
    elif args.dataset == "Tiny-ImageNet-200":
        train_loader, val_loader, test_loader, num_classes, _ = get_tinyimagenet200(args, get_train_sampler = False, transform_train = True)
        size = (3, 64, 64)
    loss_fn=torch.nn.CrossEntropyLoss()
    
    ## Loading checkpoint
    print(f"Loading checkpoint: {args.name_checkpoint}")
    checkpoint = torch.load(args.name_checkpoint, map_location=device)

    test_acc_before = checkpoint["test_acc"]
    test_loss_before = checkpoint["test_loss"]
        
    if "ResNet-18" in args.name_checkpoint:
        ## For ResNet-18, we remove one block at a time, always starting with the deepest block first, then the second deepest block, and so on, because the Wasserstein distance decreases as we move further into the architecture.
        print(f"Test acc Before: {test_acc_before: .2f}, Test Loss Before: {test_loss_before: .2f}")
        for i in range(1,8):

            if i ==1:
                ## Block 1:
                print(f"Removed block: {i: .0f}")
                if args.dataset != "Tiny-ImageNet-200":
                    model_removed = ResNet(BasicBlock, [1, 2, 2, 2], num_classes=num_classes).to(device)
                elif args.dataset == "Tiny-ImageNet-200":
                    model_removed = ResNet_orig(BasicBlock, [1, 2, 2, 2], num_classes=num_classes).to(device)
            elif i ==2:
                print(f"Removed block: {i: .0f}")
                ## Block 2:
                if args.dataset != "Tiny-ImageNet-200":
                    model_removed = ResNet(BasicBlock, [2, 1, 2, 2], num_classes=num_classes).to(device)
                elif args.dataset == "Tiny-ImageNet-200":
                    model_removed = ResNet_orig(BasicBlock, [2, 1, 2, 2], num_classes=num_classes).to(device)
            elif i ==3:
                print(f"Removed block: {i: .0f}")
                ## Block 3:
                if args.dataset != "Tiny-ImageNet-200":
                    model_removed = ResNet(BasicBlock, [2, 2, 1, 2], num_classes=num_classes).to(device)
                elif args.dataset == "Tiny-ImageNet-200":
                    model_removed = ResNet_orig(BasicBlock, [2, 2, 1, 2], num_classes=num_classes).to(device)
            elif i ==4:
                print(f"Removed block: {i: .0f}")
                ## Block 4:
                if args.dataset != "Tiny-ImageNet-200":
                    model_removed = ResNet(BasicBlock, [2, 2, 2, 1], num_classes=num_classes).to(device)
                elif args.dataset == "Tiny-ImageNet-200":
                    model_removed = ResNet_orig(BasicBlock, [2, 2, 2, 1], num_classes=num_classes).to(device)
            elif i ==5:
                print(f"Removed blocks: 3+4")
                ## 3+4:
                if args.dataset != "Tiny-ImageNet-200":
                    model_removed = ResNet(BasicBlock, [2, 2, 1, 1], num_classes=num_classes).to(device)
                elif args.dataset == "Tiny-ImageNet-200":
                    model_removed = ResNet_orig(BasicBlock, [2, 2, 1, 1], num_classes=num_classes).to(device)
            elif i ==6:
                print(f"Removed blocks: 2+3+4")
                ## 2+3+4:
                if args.dataset != "Tiny-ImageNet-200":
                    model_removed = ResNet(BasicBlock, [2, 1, 1, 1], num_classes=num_classes).to(device)
                elif args.dataset == "Tiny-ImageNet-200":
                    model_removed = ResNet_orig(BasicBlock, [2, 1, 1, 1], num_classes=num_classes).to(device)
            elif i ==7:
                print(f"Removed blocks: 1+2+3+4")
                ## 1+2+3+4:
                if args.dataset != "Tiny-ImageNet-200":
                    model_removed = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes).to(device)
                elif args.dataset == "Tiny-ImageNet-200":
                    model_removed = ResNet_orig(BasicBlock, [1, 1, 1, 1], num_classes=num_classes).to(device)            
            
            model_removed.load_state_dict(checkpoint['model_state_dict'], strict=False)

            test_acc_after, test_loss_after = val_one_epoch(model_removed, test_loader, loss_fn, device, args)
            
            print(f"Test acc After: {test_acc_after: .2f}, Test Loss After: {test_loss_after: .2f}")

            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='pytorch',
                                                        print_per_layer_stat=False, verbose=False)
                print('{:<30}  {:<8}'.format('Computational complexity (Pytorch MACs): ', macs))

                macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='aten',
                                                        print_per_layer_stat=False, verbose=False)
                print('{:<30}  {:<8}'.format('Computational complexity (ATen MACs): ', macs))
                print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            

            del model_removed
    
        
    elif "Swin-T" in args.name_checkpoint:
        
        name_csv = args.name_checkpoint.split("/")[-1].split("checkpoint_")[-1].split("_last_epoch.pt")[0]+".csv"

        df = pd.read_csv("csv/"+name_csv)
        list_of_wasserstein_distances_train = []
        list_of_wasserstein_distances_val = []

        list_of_wasserstein_distances_train = [df["block_1_train_wasserstein_cost"].values.item(),
                                                df["block_2_train_wasserstein_cost"].values.item(),
                                                df["block_3_train_wasserstein_cost"].values.item(),
                                                df["block_4_train_wasserstein_cost"].values.item(),
                                                df["block_5_train_wasserstein_cost"].values.item(),
                                                df["block_6_train_wasserstein_cost"].values.item(),
                                                df["block_7_train_wasserstein_cost"].values.item(),
                                                df["block_8_train_wasserstein_cost"].values.item(),
                                                df["block_9_train_wasserstein_cost"].values.item(),
                                                df["block_10_train_wasserstein_cost"].values.item(),
                                                df["block_11_train_wasserstein_cost"].values.item(),
                                                df["block_12_train_wasserstein_cost"].values.item()]
        
        list_of_wasserstein_distances_val = [df["block_1_val_wasserstein_cost"].values.item(),
                                                df["block_2_val_wasserstein_cost"].values.item(),
                                                df["block_3_val_wasserstein_cost"].values.item(),
                                                df["block_4_val_wasserstein_cost"].values.item(),
                                                df["block_5_val_wasserstein_cost"].values.item(),
                                                df["block_6_val_wasserstein_cost"].values.item(),
                                                df["block_7_val_wasserstein_cost"].values.item(),
                                                df["block_8_val_wasserstein_cost"].values.item(),
                                                df["block_9_val_wasserstein_cost"].values.item(),
                                                df["block_10_val_wasserstein_cost"].values.item(),
                                                df["block_11_val_wasserstein_cost"].values.item(),
                                                df["block_12_val_wasserstein_cost"].values.item()]

        list_of_divisions = [6144,  6144, 3072, 3072, 1536, 1536, 1536, 1536, 1536, 1536, 768, 768] ## CIFAR-10

        print(f"Test acc Before: {test_acc_before: .2f}, Test Loss Before: {test_loss_before: .2f}")
        mean_dist = np.mean(list_of_wasserstein_distances_val)
        std_dist = np.std(list_of_wasserstein_distances_val)
        # Calculate z-scores
        z_scores = [(i, abs((d - mean_dist) / std_dist)) 
                    for i, d in enumerate(list_of_wasserstein_distances_val)]
        
        # Calculate percentiles
        percentiles = [(i, scipy.stats.percentileofscore(list_of_wasserstein_distances_val, d)) 
                    for i, d in enumerate(list_of_wasserstein_distances_val)]
        
        # Combine metrics (lower score = more removable)
        combined_scores = [(i, z + p/100) 
                        for (i, z), (_, p) in zip(z_scores, percentiles)]

        sorted_value_index_pairs = sorted(combined_scores, key=lambda x: x[1])

        model_removed = torchvision.models.swin_t(weights = True).to(device)
        model_removed.head = torch.nn.Linear(in_features=model_removed.head.in_features, out_features=num_classes).to(device)
        model_removed.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        for index, value in sorted_value_index_pairs:
            if index==0:
                ## 1st SwinTransformerBlock
                model_removed.features[1][0] = torch.nn.Identity()

            elif index==1:
                ## 2nd SwinTransformerBlock
                model_removed.features[1][1] = torch.nn.Identity()

            elif index==2:
                ## 3rd SwinTransformerBlock
                model_removed.features[3][0] = torch.nn.Identity()
            elif index==3:
                ## 4th SwinTransformerBlock
                model_removed.features[3][1] = torch.nn.Identity()

            elif index==4:
                ## 5th SwinTransformerBlock
                model_removed.features[5][0] = torch.nn.Identity()

            elif index==5:
                ## 6th SwinTransformerBlock
                model_removed.features[5][1] = torch.nn.Identity()
            
            elif index==6:
                ## 7th SwinTransformerBlock
                model_removed.features[5][2] = torch.nn.Identity()

            elif index==7:
                ## 8th SwinTransformerBlock
                model_removed.features[5][3] = torch.nn.Identity()

            elif index==8:
                ## 9th SwinTransformerBlock
                model_removed.features[5][4] = torch.nn.Identity()

            elif index==9:
                ## 10th SwinTransformerBlock
                model_removed.features[5][5] = torch.nn.Identity()

            elif index==10:
                ## 11th SwinTransformerBlock
                model_removed.features[7][0] = torch.nn.Identity()

            elif index==11:
                ## 12th SwinTransformerBlock
                model_removed.features[7][1] = torch.nn.Identity()

            test_acc_after, test_loss_after = val_one_epoch(model_removed, test_loader, loss_fn, device, args)
            print(f"Test acc After: {test_acc_after: .2f}, Test Loss After: {test_loss_after: .2f}")

            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='pytorch',
                                                        print_per_layer_stat=False, verbose=False)
                print('{:<30}  {:<8}'.format('Computational complexity (Pytorch MACs): ', macs))

                macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='aten',
                                                        print_per_layer_stat=False, verbose=False)
                print('{:<30}  {:<8}'.format('Computational complexity (ATen MACs): ', macs))
                print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        del model_removed     


    elif "MobileNetv2" in args.name_checkpoint:
        name_csv = args.name_checkpoint.split("/")[-1].split("checkpoint_")[-1].split("_last_epoch.pt")[0]+".csv"

        df = pd.read_csv("csv/"+name_csv)
        list_of_wasserstein_distances_train = []
        list_of_wasserstein_distances_val = []

        print(f"Test acc Before: {test_acc_before: .2f}, Test Loss Before: {test_loss_before: .2f}")

        if args.dataset == "CIFAR-10":
        
            list_of_wasserstein_distances_train = [df["block_1_train_wasserstein_cost"].values.item(),
                                                    df["block_2_train_wasserstein_cost"].values.item(),
                                                    df["block_3_train_wasserstein_cost"].values.item(),
                                                    df["block_4_train_wasserstein_cost"].values.item(),
                                                    df["block_5_train_wasserstein_cost"].values.item(),
                                                    df["block_6_train_wasserstein_cost"].values.item(),
                                                    df["block_7_train_wasserstein_cost"].values.item(),
                                                    df["block_8_train_wasserstein_cost"].values.item(),
                                                    df["block_9_train_wasserstein_cost"].values.item(),
                                                    df["block_10_train_wasserstein_cost"].values.item(),
                                                    df["block_11_train_wasserstein_cost"].values.item(),
                                                    df["block_12_train_wasserstein_cost"].values.item(),
                                                    df["block_13_train_wasserstein_cost"].values.item()]
            
            list_of_wasserstein_distances_val = [df["block_1_val_wasserstein_cost"].values.item(),
                                                    df["block_2_val_wasserstein_cost"].values.item(),
                                                    df["block_3_val_wasserstein_cost"].values.item(),
                                                    df["block_4_val_wasserstein_cost"].values.item(),
                                                    df["block_5_val_wasserstein_cost"].values.item(),
                                                    df["block_6_val_wasserstein_cost"].values.item(),
                                                    df["block_7_val_wasserstein_cost"].values.item(),
                                                    df["block_8_val_wasserstein_cost"].values.item(),
                                                    df["block_9_val_wasserstein_cost"].values.item(),
                                                    df["block_10_val_wasserstein_cost"].values.item(),
                                                    df["block_11_val_wasserstein_cost"].values.item(),
                                                    df["block_12_val_wasserstein_cost"].values.item(),
                                                    df["block_13_val_wasserstein_cost"].values.item()]
        

        
            value_index_pairs = list(enumerate(list_of_wasserstein_distances_val))
            sorted_value_index_pairs = sorted(value_index_pairs, key=lambda x: x[1])

            model_removed = MobileNetV2().to(device)
            model_removed.load_state_dict(checkpoint['model_state_dict'], strict=True)

            for index, value in sorted_value_index_pairs:
                if index==1:
                    model_removed.features[2].conv[1] = torch.nn.Identity()

                elif index==2:
                    model_removed.features[3].conv[1] = torch.nn.Identity()

                elif index==3:
                    model_removed.features[5].conv[1] = torch.nn.Identity()

                elif index==4:
                    model_removed.features[6].conv[1] = torch.nn.Identity()
                
                elif index==5:
                    model_removed.features[8].conv[1] = torch.nn.Identity()

                elif index==6:
                    model_removed.features[9].conv[1] = torch.nn.Identity()

                elif index==7:
                    model_removed.features[10].conv[1] = torch.nn.Identity()

                elif index==8:
                    model_removed.features[11].conv[1] = torch.nn.Identity()

                elif index==9:
                    model_removed.features[12].conv[1] = torch.nn.Identity()

                elif index==10:
                    model_removed.features[13].conv[1] = torch.nn.Identity()

                elif index==11:
                    model_removed.features[15].conv[1] = torch.nn.Identity()

                elif index==12:
                    model_removed.features[16].conv[1] = torch.nn.Identity()
                
                elif index==13:
                    model_removed.features[17].conv[1] = torch.nn.Identity()
                

                test_acc_after, test_loss_after = val_one_epoch(model_removed, test_loader, loss_fn, device, args)
                print(f"Test acc After: {test_acc_after: .2f}, Test Loss After: {test_loss_after: .2f}")

                with torch.cuda.device(0):
                    macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='pytorch',
                                                            print_per_layer_stat=False, verbose=False)
                    print('{:<30}  {:<8}'.format('Computational complexity (Pytorch MACs): ', macs))

                    macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='aten',
                                                            print_per_layer_stat=False, verbose=False)
                    print('{:<30}  {:<8}'.format('Computational complexity (ATEN MACs): ', macs))
                    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
        else:
            list_of_wasserstein_distances_train = [df["block_1_train_wasserstein_cost"].values.item(),
                                                    df["block_2_train_wasserstein_cost"].values.item(),
                                                    df["block_3_train_wasserstein_cost"].values.item(),
                                                    df["block_4_train_wasserstein_cost"].values.item(),
                                                    df["block_5_train_wasserstein_cost"].values.item(),
                                                    df["block_6_train_wasserstein_cost"].values.item(),
                                                    df["block_7_train_wasserstein_cost"].values.item(),
                                                    df["block_8_train_wasserstein_cost"].values.item(),
                                                    df["block_9_train_wasserstein_cost"].values.item(),
                                                    df["block_10_train_wasserstein_cost"].values.item(),
                                                    df["block_11_train_wasserstein_cost"].values.item(),
                                                    df["block_12_train_wasserstein_cost"].values.item()]
            
            list_of_wasserstein_distances_val = [df["block_1_val_wasserstein_cost"].values.item(),
                                                    df["block_2_val_wasserstein_cost"].values.item(),
                                                    df["block_3_val_wasserstein_cost"].values.item(),
                                                    df["block_4_val_wasserstein_cost"].values.item(),
                                                    df["block_5_val_wasserstein_cost"].values.item(),
                                                    df["block_6_val_wasserstein_cost"].values.item(),
                                                    df["block_7_val_wasserstein_cost"].values.item(),
                                                    df["block_8_val_wasserstein_cost"].values.item(),
                                                    df["block_9_val_wasserstein_cost"].values.item(),
                                                    df["block_10_val_wasserstein_cost"].values.item(),
                                                    df["block_11_val_wasserstein_cost"].values.item(),
                                                    df["block_12_val_wasserstein_cost"].values.item()]

            value_index_pairs = list(enumerate(list_of_wasserstein_distances_val))
            sorted_value_index_pairs = sorted(value_index_pairs, key=lambda x: x[1])

            model_removed = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            model_removed.classifier[1] = torch.nn.Linear(in_features=model_removed.classifier[1].in_features, out_features=num_classes)
            model_removed.to(device)
            model_removed.load_state_dict(checkpoint['model_state_dict'], strict=True)

            for index, value in sorted_value_index_pairs:

                if index==1:
                    model_removed.features[3].conv[1] = torch.nn.Identity()

                elif index==2:
                    model_removed.features[5].conv[1] = torch.nn.Identity()

                elif index==3:
                    model_removed.features[6].conv[1] = torch.nn.Identity()
                
                elif index==4:
                    model_removed.features[8].conv[1] = torch.nn.Identity()

                elif index==5:
                    model_removed.features[9].conv[1] = torch.nn.Identity()

                elif index==6:
                    model_removed.features[10].conv[1] = torch.nn.Identity()

                elif index==7:
                    model_removed.features[11].conv[1] = torch.nn.Identity()

                elif index==8:
                    model_removed.features[12].conv[1] = torch.nn.Identity()

                elif index==9:
                    model_removed.features[13].conv[1] = torch.nn.Identity()

                elif index==10:
                    model_removed.features[15].conv[1] = torch.nn.Identity()

                elif index==11:
                    model_removed.features[16].conv[1] = torch.nn.Identity()
                
                elif index==12:
                    model_removed.features[17].conv[1] = torch.nn.Identity()


                test_acc_after, test_loss_after = val_one_epoch(model_removed, test_loader, loss_fn, device, args)
                print(f"Test acc After: {test_acc_after: .2f}, Test Loss After: {test_loss_after: .2f}")

                with torch.cuda.device(0):
                    macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='pytorch',
                                                            print_per_layer_stat=False, verbose=False)
                    print('{:<30}  {:<8}'.format('Computational complexity (Pytorch MACs): ', macs))

                    macs, params = get_model_complexity_info(model_removed, size, as_strings=True, backend='aten',
                                                            print_per_layer_stat=False, verbose=False)
                    print('{:<30}  {:<8}'.format('Computational complexity (ATEN MACs): ', macs))
                    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

                del model_removed     

    del checkpoint

if __name__ == '__main__':
    main()
