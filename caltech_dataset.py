from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
 
def load_file(directory, split):
    directory = os.path.expanduser(directory)
    file_dir = os.path.join(directory, split)
    file = open(file_dir, 'r')
    samples_to_get = file.read().splitlines()
    for f in samples_to_get:
         if f.__contains__('BACKGROUND'):
             samples_to_get.remove(f)
    file.close()
    return samples_to_get

def make_dataset(directory, class_to_idx, split):
    instances = []
    directory = os.path.expanduser(directory)
    directory = os.path.join(directory, '101_ObjectCategories')
    
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                name = target_dir + '/' + fname
                if name in split:
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)
    return instances

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split + '.txt' # This defines the split you are going to use
                                    # (split files are called 'train.txt' and 'test.txt')
        
      
        
           
        
        
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        
    def _find_classes(self, dir):
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.remove('BACKGROUND_Google')
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = ... # Provide a way to get the length (number of elements) of the dataset
        return length
