import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10 # I added it


class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    if 'MNIST' in name:
        print("MNIST dataset used")
        root = os.path.join(dset_dir, 'MNIST')
        train_kwargs = {'root':root, 'train':True, 'transform':transform, 'download':True}
        test_kwargs = {'root':root, 'train':False, 'transform':transform, 'download':True}
        dset = MNIST
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
    # I added CIFAR10 loading here
    elif 'CIFAR10' in name:
        print("CIFAR10 dataset used")
        root = os.path.join(dset_dir, 'CIFAR10')
        dset = CIFAR10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_kwargs = {'root':root, 'train':True, 'transform':transform_train, 'download':True}
        test_kwargs = {'root':root, 'train':False, 'transform': transform_test, 'download': True}

    else:
        raise UnknownDatasetError()
    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=10,
                              pin_memory=True,
                              drop_last=True)
    test_data = dset(**test_kwargs)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=10,
                             pin_memory=True,
                             drop_last=False)
    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader

    return data_loader


# You don't need to care about this part
if __name__ == '__main__':
    import argparse
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='datasets')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_loader = return_data(args)
