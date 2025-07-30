from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def Mydata(args, dir, batch_size):

    transforms_color = T.Compose([
                # T.RandomCrop(args.imageSize),    
                T.Resize([args.imageSize, args.imageSize]), 
                T.ToTensor(),
            ])  

    transforms_gray = T.Compose([ 
                T.Grayscale(num_output_channels=1),
                # T.RandomCrop(args.imageSize),
                T.Resize([args.imageSize, args.imageSize]), 
                T.ToTensor(),
            ]) 
    
    if args.channel_cover == 1:  
        transforms_image = transforms_gray
    else:
        transforms_image = transforms_color

    if args.channel_secret == 1:  
        transforms_image = transforms_gray
    else:
        transforms_image = transforms_color

    dataset = ImageFolder(dir, transforms_image)
    
    assert dataset 

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )   

    return dataloader 
