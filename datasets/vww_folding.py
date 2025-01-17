import pyvww
from torchvision import transforms
import ai8x

#folder paths to images and annotations
image_path = "/home/bobbycounts/visualWakeNew/ai8x-training/datasets/vww/all/"
ann_path = "/home/bobbycounts/visualWakeNew/ai8x-training/datasets/vww/annotations_vww/"


def vww_get_datasets(data, load_train=True, load_test=True):

    (data_dir, args) = data
    
    resolution = (256,256)
    resolution1 = (270,270)
    
    #set which dataset to use

    dataset = pyvww.pytorch.datasets.VisualWakeWordsClassificationFolding

    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize(resolution1),
            transforms.RandomCrop(resolution),
            #transforms.ColorJitter(0.1,0.1,0.1,0.1),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(degrees=(0, 45)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        
        train_dataset = dataset(root=image_path, annFile=ann_path + "instances_train.json", transform=train_transform)
        
    else:
        train_dataset = None
        
        
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = dataset(root=image_path, annFile=ann_path + "instances_val.json", transform=test_transform)
        
    else:
        test_dataset = None
        
    return train_dataset, test_dataset

datasets = [
    {
        'name': 'vww_folding',
        'input': (48, 64, 64),
        'output': ('false', 'true'),
        'loader': vww_get_datasets,
    },
]








