"""
Krushi Mitra AI — Plant Disease Model Training Script
======================================================
Model   : ResNet9 (custom, same as Harvestify / disease_utils.py)
Dataset : New Plant Diseases Dataset (Kaggle: vipoooool/new-plant-diseases-dataset)
Output  : models/plant_disease_model.pth

FOLDER STRUCTURE EXPECTED (after extracting the Kaggle zip):
  Krushi-Mitra/
    New Plant Diseases Dataset/
      train/
        Apple___Apple_scab/  ...
      valid/
        Apple___Apple_scab/  ...

USAGE
-----
1. Download & extract dataset from Kaggle manually
2. Run:  python train_disease_model.py

REQUIREMENTS
------------
pip install torch torchvision tqdm
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
# Update this path to wherever you extracted the zip
DATASET_ROOT  = "./New Plant Diseases Dataset"
TRAIN_DIR     = os.path.join(DATASET_ROOT, "train")
VAL_DIR       = os.path.join(DATASET_ROOT, "valid")
MODEL_OUT     = "./models/plant_disease_model.pth"

BATCH_SIZE    = 16          # safe for 4GB VRAM (GTX 1650)
EPOCHS        = 10
LR            = 1e-3
IMG_SIZE      = 224         # slightly reduced to fit in 4GB VRAM
NUM_WORKERS   = 0           # IMPORTANT: keep 0 on Windows to avoid multiprocessing errors
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU          : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

os.makedirs("./models", exist_ok=True)

# ─────────────────────────────────────────────
# 2. VERIFY DATASET PATH
# ─────────────────────────────────────────────
if not os.path.exists(TRAIN_DIR):
    print(f"\n[ERROR] Train folder not found at: {TRAIN_DIR}")
    print("Please extract the Kaggle zip and update DATASET_ROOT in this script.")
    print("Expected structure:")
    print("  New Plant Diseases Dataset/")
    print("    train/  <-- 38 class folders here")
    print("    valid/  <-- 38 class folders here")
    exit(1)

# ─────────────────────────────────────────────
# 3. CANONICAL CLASS ORDER (must match disease_utils.py exactly)
# ─────────────────────────────────────────────
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]
NUM_CLASSES   = len(DISEASE_CLASSES)   # 38
CLASS_TO_IDX  = {cls: idx for idx, cls in enumerate(DISEASE_CLASSES)}

# ─────────────────────────────────────────────
# 4. CUSTOM DATASET — remaps folder labels to canonical order
# ─────────────────────────────────────────────
class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []

        folder_dataset = datasets.ImageFolder(root_dir)
        folder_classes = folder_dataset.classes

        skipped = 0
        for path, folder_idx in folder_dataset.samples:
            folder_class = folder_classes[folder_idx]
            if folder_class in CLASS_TO_IDX:
                self.samples.append((path, CLASS_TO_IDX[folder_class]))
            else:
                skipped += 1

        if skipped:
            print(f"  [WARN] Skipped {skipped} images from unrecognised classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ─────────────────────────────────────────────
# 5. TRANSFORMS
# ─────────────────────────────────────────────
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 6. DATA LOADERS
# ─────────────────────────────────────────────
print("\n[INFO] Loading datasets...")
train_dataset = PlantDataset(TRAIN_DIR, transform=train_tfm)
val_dataset   = PlantDataset(VAL_DIR,   transform=val_tfm)

print(f"[INFO] Train images : {len(train_dataset)}")
print(f"[INFO] Val images   : {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)

# ─────────────────────────────────────────────
# 7. MODEL — ResNet9
# ─────────────────────────────────────────────
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1  = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2  = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # works for any input size (224 or 256)
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

model = ResNet9(3, NUM_CLASSES).to(DEVICE)
print(f"[INFO] ResNet9 ready — {NUM_CLASSES} classes\n")

# ─────────────────────────────────────────────
# 8. LOSS / OPTIMIZER / SCHEDULER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
)

# ─────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────
best_val_acc = 0.0

print("=" * 60)
print(f"  Training ResNet9  |  {EPOCHS} epochs  |  device: {DEVICE}")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for images, labels in tqdm(train_loader,
                                desc=f"Epoch {epoch:>2}/{EPOCHS} [Train]",
                                leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    # ── Validate ───────────────────────────
    model.eval()
    val_loss = 0.0
    correct  = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader,
                                    desc=f"Epoch {epoch:>2}/{EPOCHS} [Val]  ",
                                    leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds    = outputs.argmax(dim=1)
            correct  += (preds == labels).sum().item()

    val_loss /= len(val_dataset)
    val_acc   = correct / len(val_dataset) * 100
    elapsed   = time.time() - t0

    print(f"Epoch {epoch:>2}/{EPOCHS}  |  "
          f"Train Loss: {train_loss:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  |  "
          f"Val Acc: {val_acc:.2f}%  |  "
          f"{elapsed:.1f}s")

    # ── Save best ──────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ✅ Best model saved → {MODEL_OUT}  (val_acc={val_acc:.2f}%)")

print("\n" + "=" * 60)
print(f"  Done!  Best Val Accuracy : {best_val_acc:.2f}%")
print(f"  Model saved to           : {MODEL_OUT}")
print("=" * 60)

# ─────────────────────────────────────────────
# 10. VERIFY SAVED MODEL
# ─────────────────────────────────────────────
print("\n[INFO] Verifying saved model...")
chk = ResNet9(3, NUM_CLASSES)
chk.load_state_dict(torch.load(MODEL_OUT, map_location="cpu"))
chk.eval()
print("[INFO] ✅ Model verified — ready for Krushi Mitra AI!")
print(f"[INFO] Place '{MODEL_OUT}' at 'models/plant_disease_model.pth' in your project.")
