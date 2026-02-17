from torchvision import datasets, transforms

from torch.utils.data import DataLoader


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

dataset = datasets.ImageFolder(root="./brain_tumor_dataset", transform=transform)

loader = DataLoader(dataset, batch_size=32, shuffle=True)


    
