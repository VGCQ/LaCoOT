import torch
import torchvision
from torchvision import transforms
import os


def get_tinyimagenet200(args, get_train_sampler = False, transform_train = True):
    
    DATA_DIR = args.root
    TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(VALID_DIR, transform=transform_test)
    test_dataset = torchvision.datasets.ImageFolder(TEST_DIR, transform=transform_test)

    dataset_labels = train_dataset.classes
    num_classes = len(train_dataset.classes)
    
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
