import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

DATA_ROOT = os.environ.get("TOMATO_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Tomatoes Dataset"))
OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]
IMG_SIZE = 128
BATCH = 64
EPOCHS = 20
SEED = 42

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=train_tf)
val_ds = datasets.ImageFolder(f"{DATA_ROOT}/val", transform=val_tf)
print("classes (ImageFolder order):", train_ds.classes)
print("train size:", len(train_ds), "val size:", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(train_ds.classes))
    return m


model = build_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
t_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    scheduler.step()
    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            all_pred += pred.cpu().tolist()
            all_true += y.cpu().tolist()
    val_loss = running_loss / total
    val_acc = correct / total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{OUT}/cnn_best.pt")
        best_pred, best_true = all_pred, all_true

total_train_time = time.time() - t_start
print(f"\nTotal training time: {total_train_time:.1f}s for {EPOCHS} epochs "
      f"({total_train_time/EPOCHS:.1f}s/epoch)")
print(f"Best val accuracy: {best_val_acc:.4f}")

cm = confusion_matrix(best_true, best_pred)
f1 = f1_score(best_true, best_pred, average="macro")
print("Confusion matrix (rows=true, cols=pred), class order:", train_ds.classes)
print(cm)
print("Macro F1 at best epoch:", f1)

with open(f"{OUT}/cnn_history.json", "w") as f:
    json.dump({
        "history": history,
        "classes_imagefolder_order": train_ds.classes,
        "best_val_acc": best_val_acc,
        "best_val_f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "total_train_seconds": total_train_time,
        "epochs": EPOCHS,
        "device": str(device),
    }, f, indent=2)
print("Saved cnn_history.json and cnn_best.pt")
