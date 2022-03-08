from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
from pyvww.utils import VisualWakeWords
import torch

class VisualWakeWordsClassification(VisionDataset):
    """`Visual Wake Words <https://arxiv.org/abs/1906.05721>`_ Dataset.
    Args:
        root (string): Root directory where COCO images are downloaded to.
        annFile (string): Path to json visual wake words annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(VisualWakeWordsClassification, self).__init__(root, transforms, transform, target_transform)
        self.vww = VisualWakeWords(annFile)
        self.ids = list(sorted(self.vww.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the index of the target class.
        """
        vww = self.vww
        img_id = self.ids[index]
        ann_ids = vww.getAnnIds(imgIds=img_id)
        if ann_ids:
            target = vww.loadAnns(ann_ids)[0]['category_id']
        else:
            target = 0

        path = vww.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)
        
        
        
class VisualWakeWordsClassificationFolding(VisionDataset):
    """`Visual Wake Words <https://arxiv.org/abs/1906.05721>`_ Dataset.
    Args:
        root (string): Root directory where COCO images are downloaded to.
        annFile (string): Path to json visual wake words annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(VisualWakeWordsClassificationFolding, self).__init__(root, transforms, transform, target_transform)
        self.vww = VisualWakeWords(annFile)
        self.ids = list(sorted(self.vww.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the index of the target class.
        """
        vww = self.vww
        img_id = self.ids[index]
        ann_ids = vww.getAnnIds(imgIds=img_id)
        if ann_ids:
            target = vww.loadAnns(ann_ids)[0]['category_id']
        else:
            target = 0

        path = vww.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)


        #test pattern
        #img = torch.tensor(range(1,193))
        #img = torch.reshape(img, (3,8,8))

        #implement folding of input data
        alpha = 4
        c, h, w = img.size()
        
        assert (h/alpha).is_integer(), "h / alpha must be an integer!"
        assert (w/alpha).is_integer(), "w / alpha must be an integer!"
        
        folded_tensor = torch.zeros((alpha*alpha*c, h//alpha, w//alpha))
        
        tensor_channel = 0
       
        for c_idx in range(c):
            img_single_channel = img[c_idx, :, :]
            for row_start in range(alpha):
                for col_start in range(alpha):
                    folded_tensor[tensor_channel, :, :] = img_single_channel[row_start::alpha, col_start::alpha]
                    tensor_channel += 1          

        return folded_tensor, target


    def __len__(self):
        return len(self.ids)
