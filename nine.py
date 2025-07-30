print("Running Image Classification Script...")

# === Import Dependencies ===
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# === STEP 1: Data Loading ===

data_dir = r"C:\Users\abhishek\Downloads\dataset_structured\dataset\train"

if not os.path.exists(data_dir):
    print("❌ Dataset path is incorrect!")
else:
    print("✅ Found dataset folder:", data_dir)
    print("Subfolders (classes):", os.listdir(data_dir))

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Classes: {dataset.classes}")
print(f"Total Images: {len(dataset)}")

# === STEP 2: Load Pretrained ResNet18 and Modify ===

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))  # Binary output for cats/dogs

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === STEP 3: Loss and Optimizer ===

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# === STEP 4: Train the Model ===

for epoch in range(5):  # You can try 10+ for even better results
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {running_loss / len(dataloader):.4f}")

# === STEP 5: Evaluate ===

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# === STEP 6: Predict on Test Images ===

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_folder = "test_images"

print("\nPredictions on test images:")
if not os.path.exists(test_folder):
    print("⚠️ Test folder 'test_images' not found!")
else:
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(test_folder, filename)
            image = Image.open(path).convert('RGB')
            image_tensor = transform_test(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                _, pred = torch.max(output, 1)
                predicted_class = dataset.classes[pred.item()]
                print(f"{filename} → {predicted_class}")

            plt.imshow(image)
            plt.title(f"{filename}: {predicted_class}")
            plt.axis('off')
            plt.show()

# === STEP 7: Save the Model ===

torch.save(model.state_dict(), "cat_dog_model.pth")
print("✅ Model saved as cat_dog_model.pth")
