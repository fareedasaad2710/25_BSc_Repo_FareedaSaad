from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T  # ✅ Needed for transforms
import torch

# Define Dataset
class KittiDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, self.images[idx], label_path

# Initialize Models
yolo = YOLODetector()
depth = DepthEstimator()
fusion = FusionHead()
optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()  # Replace with better 3D loss later

# Dataset and DataLoader
ds = KittiDataset(f"{BASE_DIR}/images/train", f"{BASE_DIR}/labels/train")
loader = DataLoader(ds, batch_size=1, shuffle=True)

print(f"Total training images: {len(ds)}")

def parse_label_file(label_path):
    targets = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            if parts[0] == "DontCare":
                continue
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            targets.append(torch.tensor([x, y, z]))
    return targets

# Training Loop
for epoch in range(2):  # You can increase this later
    print(f"\n--- Epoch {epoch+1} ---")
    for image, name, label_path in loader:
        image = image[0]  # (C, H, W)
        pil_image = T.ToPILImage()(image)
        print(f"Processing image: {name[0]}")

        boxes, _, _ = yolo.detect(pil_image)
        depth_map = depth.predict(pil_image)

        if len(boxes) == 0:
            print(f"[{epoch}] {name[0]} | No detections, skipping")
            continue

        feats = extract_box_depth_features(boxes, depth_map)
        preds = fusion(feats)

        label_targets = parse_label_file(label_path[0])

        if len(label_targets) < len(preds):
            print(f"[{epoch}] {name[0]} | More predictions than labels, skipping")
            continue

        # 🚨 TEMPORARY: naive assignment (you can improve this using IoU later)
        targets = torch.stack(label_targets[:len(preds)])

        loss = criterion(preds, targets)
        loss.backward()

        print(f"Preds: {preds[0].detach().cpu().numpy()}, Targets: {targets[0].cpu().numpy()}")

        optimizer.step()
        optimizer.zero_grad()
        print(f"[{epoch}] {name[0]} | Boxes: {len(boxes)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} done")

# ✅ Save model after training
torch.save(fusion.state_dict(), "fusion_head.pth")
print("Fusion model saved!")
