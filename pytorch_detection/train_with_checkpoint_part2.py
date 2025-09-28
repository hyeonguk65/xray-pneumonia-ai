import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from rsna_dataset import RSNADataset
from model import get_faster_rcnn

# 1. 하이퍼파라미터
num_classes = 2    
num_epochs = 10    
batch_size = 2     
lr = 0.005         

# 2. Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

# 3. Dataset & DataLoader 
transform = T.Compose([T.ToTensor()])

dataset = RSNADataset(
    img_dir=r"C:\Users\user\Downloads\rsna-pneumonia-detection-challenge\stage_2_train_images",
    csv_file=r"C:\Users\user\Downloads\rsna-pneumonia-detection-challenge\stage_2_train_labels.csv",
    transforms=transform,
    resize=512
)

def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, collate_fn=collate_fn
)

# 4. 모델 및 옵티마이저
model = get_faster_rcnn(num_classes=num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

# 5. 저장된 체크포인트 불러오기 
checkpoint = torch.load("checkpoint\checkpoint_epoch8.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]  

print(f"{start_epoch} epoch까지 학습된 모델 불러옴")

# 6. 이어서 학습 
for epoch in range(start_epoch, num_epochs): 
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):  
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        # 중간 로그 (예: 500 step마다 출력)
        if batch_idx % 500 == 0:
            print(f"Epoch {epoch+1}, Step {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")


    # 에포크 끝난 후 로그
    print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

    # epoch마다 체크포인트 저장
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss,
    }, f"checkpoint/checkpoint_epoch{epoch+1}.pth")


print("10 epoch 학습 완료 최종 모델 저장됨")
