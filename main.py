# =========================
# Brain Tumor CNN Training
# =========================

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# -------------------------
# 1. Device Configuration
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------
# 2. Image Transformations
# -------------------------
# Resize all images to same size and convert to tensor
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# -------------------------
# 3. Load Dataset
# -------------------------
# Folder structure:
# brain_tumor_dataset/
#    ├── yes/
#    └── no/
dataset = datasets.ImageFolder(root="./brain_tumor_dataset", transform=transform)

# -------------------------
# 4. Train/Test Split
# -------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -------------------------
# 5. CNN Model Definition
# -------------------------
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # First Conv Block
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second Conv Block
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # After two MaxPool layers:
        # 224 -> 112 -> 56
        self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(32 * 56 * 56, 1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# -------------------------
# 6. Initialize Model
# -------------------------
torch.manual_seed(42)

model = BrainTumorCNN().to(device)

# BCEWithLogitsLoss combines Sigmoid + BCELoss (better stability)
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# 7. Training Loop
# -------------------------
epochs = 10

for epoch in range(epochs):

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        preds = torch.sigmoid(outputs)
        predicted = (preds > 0.5).float()

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

# -------------------------
# 8. Evaluation on Test Set
# -------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        preds = torch.sigmoid(outputs)
        predicted = (preds > 0.5).float()

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"\nTest Accuracy: {test_accuracy:.4f}")
