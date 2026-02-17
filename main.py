from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from torch import nn

# from sklearn.preprocessing import StandardScaler

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

dataset = datasets.ImageFolder(root="./brain_tumor_dataset", transform=transform)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in loader:
    print(images.shape)
    print(labels)
    break
# scaler = StandardScaler()
# images = scaler.fit(images)

images = torch.tensor(images)
labels = torch.tensor(labels)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


class AI_Brain_Tumar_Dataset(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=18),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=18, out_channels=6),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=6, out_channels=1),
        )

    def forward(self, X):
        self.model(X)

torch.manual_seed(42)
model12 = AI_Brain_Tumar_Dataset()

