import os
from utils import *
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
])


class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.files = os.listdir(os.path.join(path, 'SegmentationClass'))
    
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        segment_file = self.files[idx]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_file)
        image_path = os.path.join(self.path, 'JPEGImages', segment_file.replace('png', 'jpg'))
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)
    
    
if __name__ == '__main__':
    dataset = MyDataset('UNet/VOCdevkit/VOC2012')
    print(dataset[0][0].shape, dataset[0][1].shape)