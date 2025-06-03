import os
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import tkinter as tk

# Dataset
class CatsDogsDataset(Dataset):
    def __init__(self, folder_path, transform=None, return_raw=False):
        self.folder_path = folder_path
        self.transform = transform
        self.return_raw = return_raw
        self.images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        if img_name.lower().startswith('gato'):
            label = 0
        elif img_name.lower().startswith('perro'):
            label = 1
        else:
            raise ValueError(f"Archivo con nombre inesperado: {img_name}")

        if self.return_raw:
            return image, image_tensor, label, img_name
        else:
            return image_tensor, label

# Config
BATCH_SIZE = 32
IMG_SIZE = 224  # aumentar tamaño
DATA_DIR = "./train"

# Data augmentation para entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalización ImageNet estándar
                         std=[0.229, 0.224, 0.225]),
])

# Para validación y GUI solo resize + normalización
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset y loaders
full_dataset = CatsDogsDataset(DATA_DIR, transform=train_transform, return_raw=False)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Cambiar transform para val_dataset
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Modelo mejorado con BatchNorm y Dropout
class CNNImproved(nn.Module):
    def __init__(self):
        super(CNNImproved, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * (IMG_SIZE // 16) * (IMG_SIZE // 16), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNImproved()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # lr un poco más bajo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"[{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")

    # Validación rápida cada epoch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Val Accuracy: {100 * correct / total:.2f}%")

# Preparar GUI dataset con transform val_transform + return_raw=True
gui_dataset = CatsDogsDataset(DATA_DIR, transform=val_transform, return_raw=True)
all_data = [gui_dataset[i] for i in range(len(gui_dataset))]

softmax = nn.Softmax(dim=1)
def get_probabilities(image_tensor):
    model.eval()
    with torch.no_grad():
        inp = image_tensor.unsqueeze(0).to(device)
        logits = model(inp)
        probs = softmax(logits).cpu().squeeze(0)
        return probs.numpy()

class ImageBrowser:
    def __init__(self, master, data_list):
        self.master = master
        self.master.title("Clasificador Gato vs Perro Mejorado")
        self.data_list = data_list
        self.index = 0

        self.img_label = tk.Label(master)
        self.img_label.pack(padx=10, pady=10)

        self.prob_label = tk.Label(master, text="", font=("Arial", 14))
        self.prob_label.pack(pady=(0,10))

        btn_frame = tk.Frame(master)
        btn_frame.pack()

        self.prev_btn = tk.Button(btn_frame, text="Anterior", command=self.prev_image)
        self.prev_btn.grid(row=0, column=0, padx=5)

        self.next_btn = tk.Button(btn_frame, text="Siguiente", command=self.next_image)
        self.next_btn.grid(row=0, column=1, padx=5)

        self.update_display()

    def update_display(self):
        pil_img, img_tensor, true_label, img_name = self.data_list[self.index]
        disp_img = pil_img.resize((256, 256))
        tk_img = ImageTk.PhotoImage(disp_img)

        self.img_label.configure(image=tk_img)
        self.img_label.image = tk_img

        probs = get_probabilities(img_tensor)
        pct_gato = probs[0] * 100
        pct_perro = probs[1] * 100

        texto = f"{img_name}\nProb Gato: {pct_gato:.2f}%   Prob Perro: {pct_perro:.2f}%"
        self.prob_label.configure(text=texto)

        self.prev_btn.configure(state=("normal" if self.index > 0 else "disabled"))
        self.next_btn.configure(state=("normal" if self.index < len(self.data_list)-1 else "disabled"))

    def next_image(self):
        if self.index < len(self.data_list) - 1:
            self.index += 1
            self.update_display()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.update_display()

root = tk.Tk()
app = ImageBrowser(root, all_data)
root.mainloop()
