import torch
import torchvision
from torchvision import transforms


def get_cifar10(args, size_train_set = 0.9, get_train_sampler = False, transform_train = True):
    
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32,4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = torchvision.datasets.CIFAR10(root=args.root,
                                                train=True, 
                                                transform=transform,
                                                download=False)

    train_size = int(size_train_set * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    test_dataset = torchvision.datasets.CIFAR10(root=args.root,
                                                train=False, 
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                                                download=False)

    dataset_labels = test_dataset.classes
    num_classes = len(test_dataset.classes)
    
    train_sampler = None
    validation_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None and transform_train),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   persistent_workers=args.workers > 0)

    valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True, sampler=validation_sampler,
                                                   persistent_workers=args.workers > 0)
    
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True, sampler=test_sampler,
                                                  persistent_workers=args.workers > 0)
    
    if(get_train_sampler):
        return train_dataloader, valid_dataloader, test_dataloader, train_sampler, num_classes, dataset_labels
    
    return train_dataloader, valid_dataloader, test_dataloader, num_classes, dataset_labels
