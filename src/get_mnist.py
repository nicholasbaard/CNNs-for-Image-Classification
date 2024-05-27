from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist(data_dir="../data", batch_size=32):

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    test_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])
    
    train_dataset = datasets.MNIST(root=data_dir,
                                    train=True, 
                                    transform=train_transforms, 
                                    download=True
                                    )
    
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    drop_last=True
                                    )
    
    test_dataset = datasets.MNIST(root=data_dir,
                                    train=True, 
                                    transform=test_transforms, 
                                    download=True
                                    )
    
    test_dataloader = DataLoader(test_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    drop_last=True
                                    )
    
    # for image, label in train_dataloader:
    #     print(image.shape)
    #     print(label.shape)
    #     break

    return train_dataloader, test_dataloader

if __name__ == "__main__":

    train_dataloader, test_dataloader = get_mnist()

    
