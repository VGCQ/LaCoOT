import os 
import argparse
import torch
import torchvision
import random
import numpy as np
import re
import wandb

from models.ResNet import ResNet, BasicBlock, ResNet_orig
from models.Mobilenetv2 import MobileNetV2
from utils import SaveOutput, SaveInput
from train import train_one_epoch_wasserstein, val_one_epoch_wasserstein, train_one_epoch_wasserstein_2, val_one_epoch_wasserstein_2
from dataloaders.cifar10 import get_cifar10
from dataloaders.tinyimagenet200 import get_tinyimagenet200


def main():
    parser = argparse.ArgumentParser(description='LaCoOT')
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
    parser.add_argument('--name_run', default=None, help='Name of run in wandb (default:None ->Automatic)')
    parser.add_argument('--entity', default="None", help='Name of wandb entity (default:"None")')
    
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

    ## Function
    if args.model == "Swin-T" or args.model == "MobileNetv2":
        def train_one_epoch_w(model, train_loader, optimizer, loss_fn, hooks, save_input, save_output, device, args):
            return(train_one_epoch_wasserstein_2(model, train_loader, optimizer, loss_fn, hooks, save_input, save_output, device, args))
        def val_one_epoch_w(model, val_loader, loss_fn, hooks, save_input, save_output, device, args):
            return(val_one_epoch_wasserstein_2(model, val_loader, loss_fn, hooks, save_input, save_output, device, args))
    elif args.model == "ResNet-18":
        def train_one_epoch_w(model, train_loader, optimizer, loss_fn, hooks, save_input, save_output, device, args):
            return(train_one_epoch_wasserstein(model, train_loader, optimizer, loss_fn, hooks, save_output, device, args))
        def val_one_epoch_w(model, val_loader, loss_fn, hooks, save_input, save_output, device, args):
            return(val_one_epoch_wasserstein(model, val_loader, loss_fn, hooks, save_output, device, args))
    
    ## DATASET:
    if args.dataset == "CIFAR-10":
        train_loader, val_loader, test_loader, num_classes, _ = get_cifar10(args, size_train_set = 0.9, get_train_sampler = False, transform_train = True)
    elif args.dataset == "Tiny-ImageNet-200":
        train_loader, val_loader, test_loader, num_classes, _ = get_tinyimagenet200(args, get_train_sampler = False, transform_train = True)
    
    ## MODEL:
    if args.model == "ResNet-18" and args.dataset != "Tiny-ImageNet-200":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)

    elif args.model == "ResNet-18" and args.dataset == "Tiny-ImageNet-200":
        model = ResNet_orig(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)

    elif args.model == "MobileNetv2" and args.dataset == "CIFAR-10":
        model = MobileNetV2().to(device)

    elif args.model == "MobileNetv2" and args.dataset != "CIFAR-10":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
        model.to(device)

    elif args.model == "Swin-T":
        model = torchvision.models.swin_t(weights = True).to(device)
        model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=num_classes).to(device)
    
    if args.model == "MobileNetv2":
        for name, module in model.named_modules():
            if type(module) == torch.nn.ReLU6: ## MobileNetv2
                change_name= re.sub(r'\.(\d+)', r'[\1]', name)
                exec(f'model.{change_name} = torch.nn.ReLU6(inplace=False)')
    elif args.model == "ResNet-18":
        for name, module in model.named_modules():
            if type(module) == torch.nn.ReLU: ## ResNet-18
                change_name= re.sub(r'\.(\d+)', r'[\1]', name)
                exec(f'model.{change_name} = torch.nn.ReLU(inplace=False)')
    
    ## Hook 
    ######################
    ######################
    ## HERE DEPENDING ON THE MODEL, CHOOSE THE RIGHT BLOCK TO MONITOR
    save_output = SaveOutput()
    save_input = SaveInput()
    hooks = {}

    if "ResNet" in args.model :
        for name, module in model.named_modules():
            if "relu2" in name: ## ResNet
                hooks[name] = module.register_forward_hook(save_output)

    if args.model == "Swin-T":
        hooks={}

        hooks["features.1.0.norm1"] = model.features[1][0].norm1.register_forward_hook(save_input)
        hooks["features.1.0.mlp.4"] = model.features[1][0].mlp[4].register_forward_hook(save_output)

        hooks["features.1.1.norm1"] = model.features[1][1].norm1.register_forward_hook(save_input)
        hooks["features.1.1.mlp.4"] = model.features[1][1].mlp[4].register_forward_hook(save_output)

        hooks["features.3.0.norm1"] = model.features[3][0].norm1.register_forward_hook(save_input)
        hooks["features.3.0.mlp.4"] = model.features[3][0].mlp[4].register_forward_hook(save_output)

        hooks["features.3.1.norm1"] = model.features[3][1].norm1.register_forward_hook(save_input)
        hooks["features.3.1.mlp.4"] = model.features[3][1].mlp[4].register_forward_hook(save_output)

        hooks["features.5.0.norm1"] = model.features[5][0].norm1.register_forward_hook(save_input)
        hooks["features.5.0.mlp.4"] = model.features[5][0].mlp[4].register_forward_hook(save_output)

        hooks["features.5.1.norm1"] = model.features[5][1].norm1.register_forward_hook(save_input)
        hooks["features.5.1.mlp.4"] = model.features[5][1].mlp[4].register_forward_hook(save_output)

        hooks["features.5.2.norm1"] = model.features[5][2].norm1.register_forward_hook(save_input)
        hooks["features.5.2.mlp.4"] = model.features[5][2].mlp[4].register_forward_hook(save_output)

        hooks["features.5.3.norm1"] = model.features[5][3].norm1.register_forward_hook(save_input)
        hooks["features.5.3.mlp.4"] = model.features[5][3].mlp[4].register_forward_hook(save_output)

        hooks["features.5.4.norm1"] = model.features[5][4].norm1.register_forward_hook(save_input)
        hooks["features.5.4.mlp.4"] = model.features[5][4].mlp[4].register_forward_hook(save_output)

        hooks["features.5.5.norm1"] = model.features[5][5].norm1.register_forward_hook(save_input)
        hooks["features.5.5.mlp.4"] = model.features[5][5].mlp[4].register_forward_hook(save_output)

        hooks["features.7.0.norm1"] = model.features[7][0].norm1.register_forward_hook(save_input)
        hooks["features.7.0.mlp.4"] = model.features[7][0].mlp[4].register_forward_hook(save_output)

        hooks["features.7.1.norm1"] = model.features[7][1].norm1.register_forward_hook(save_input)
        hooks["features.7.1.mlp.4"] = model.features[7][1].mlp[4].register_forward_hook(save_output)

    if args.model == "MobileNetv2":
        hooks={}

        if args.dataset == "CIFAR-10":
            hooks["features.2.conv.1_in"] = model.features[2].conv[1].register_forward_hook(save_input)
            hooks["features.2.conv.1_out"] = model.features[2].conv[1].register_forward_hook(save_output)

        hooks["features.3.conv.1_in"] = model.features[3].conv[1].register_forward_hook(save_input)
        hooks["features.3.conv.1_out"] = model.features[3].conv[1].register_forward_hook(save_output)

        hooks["features.5.conv.1_in"] = model.features[5].conv[1].register_forward_hook(save_input)
        hooks["features.5.conv.1_out"] = model.features[5].conv[1].register_forward_hook(save_output)

        hooks["features.6.conv.1_in"] = model.features[6].conv[1].register_forward_hook(save_input)
        hooks["features.6.conv.1_out"] = model.features[6].conv[1].register_forward_hook(save_output)
        
        hooks["features.8.conv.1_in"] = model.features[8].conv[1].register_forward_hook(save_input)
        hooks["features.8.conv.1_out"] = model.features[8].conv[1].register_forward_hook(save_output)

        hooks["features.9.conv.1_in"] = model.features[9].conv[1].register_forward_hook(save_input)
        hooks["features.9.conv.1_out"] = model.features[9].conv[1].register_forward_hook(save_output)
        
        hooks["features.10.conv.1_in"] = model.features[10].conv[1].register_forward_hook(save_input)
        hooks["features.10.conv.1_out"] = model.features[10].conv[1].register_forward_hook(save_output)
        
        hooks["features.11.conv.1_in"] = model.features[11].conv[1].register_forward_hook(save_input)
        hooks["features.11.conv.1_out"] = model.features[11].conv[1].register_forward_hook(save_output)
        
        hooks["features.12.conv.1_in"] = model.features[12].conv[1].register_forward_hook(save_input)
        hooks["features.12.conv.1_out"] = model.features[12].conv[1].register_forward_hook(save_output)
        
        hooks["features.13.conv.1_in"] = model.features[13].conv[1].register_forward_hook(save_input)
        hooks["features.13.conv.1_out"] = model.features[13].conv[1].register_forward_hook(save_output)
        
        hooks["features.15.conv.1_in"] = model.features[15].conv[1].register_forward_hook(save_input)
        hooks["features.15.conv.1_out"] = model.features[15].conv[1].register_forward_hook(save_output)
        
        hooks["features.16.conv.1_in"] = model.features[16].conv[1].register_forward_hook(save_input)
        hooks["features.16.conv.1_out"] = model.features[16].conv[1].register_forward_hook(save_output)
        
        hooks["features.17.conv.1_in"] = model.features[17].conv[1].register_forward_hook(save_input)
        hooks["features.17.conv.1_out"] = model.features[17].conv[1].register_forward_hook(save_output)

    ## Loss/optimizer/scheduler
    loss_fn=torch.nn.CrossEntropyLoss()
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        
    wandb.init(project="LaCoOT", entity=args.entity)
    
    if args.name_run != None:
        name_run = args.name_run
    else:
        name_run = args.dataset+"_"+args.model+"_lambda_"+str(args.lambda_reg)

    wandb.run.name = name_run
    wandb.config.lambda_reg = args.lambda_reg

    ## TRAINING:
    for epoch in range(1,args.epochs+1):
        train_acc, train_loss, cross_loss, mean_wasserstein_cost, wasserstein_cost_each_layer = train_one_epoch_w(model, train_loader, optimizer, loss_fn, hooks, save_input, save_output, device, args)
        val_acc, val_loss, val_cross_loss, val_mean_wasserstein_cost, val_wasserstein_cost_each_layer = val_one_epoch_w(model, val_loader, loss_fn, hooks, save_input, save_output, device, args)
        test_acc, test_loss, _, _, _ = val_one_epoch_w(model, test_loader, loss_fn, hooks, save_input, save_output, device, args)
        
        wandb.log(
                {"train_acc": train_acc, "train_loss": train_loss, "cross_loss":cross_loss, "mean_wasserstein_cost":mean_wasserstein_cost,
                "val_acc": val_acc, "val_loss": val_loss, "val_cross_loss":val_cross_loss, "val_mean_wasserstein_cost":val_mean_wasserstein_cost,
                "test_acc": test_acc, "test_loss": test_loss,
                "lr":optimizer.param_groups[0]['lr']}, step=epoch)
        
        for l in range(len(hooks.keys())-1):
            wandb.log({"wasserstein_cost_each_layer/"+str(list(hooks.keys())[l]): wasserstein_cost_each_layer[l],
                    "val_wasserstein_cost_each_layer/"+str(list(hooks.keys())[l]): val_wasserstein_cost_each_layer[l]},
                    step=epoch)

        torch.save({               
                'epoch': epoch,
                'train_loss':train_loss, 'train_acc':train_acc,
                'val_loss':val_loss, 'val_acc':val_acc,
                'test_loss':test_loss, 'test_acc':test_acc, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},os.path.join(args.dir_to_save_checkpoint,"checkpoint_"+name_run+"_last_epoch.pt"))
        
        if args.optimizer == "SGD":
            scheduler.step()
    
    wandb.finish()
   
if __name__ == '__main__':
    main()
