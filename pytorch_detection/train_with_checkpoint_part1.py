import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from rsna_dataset import RSNADataset
from model import get_faster_rcnn

# 1. 하이퍼파라미터
num_classes = 2    # NORMAL / PNEUMONIA
num_epochs = 10    
batch_size = 2     
lr = 0.005         # torchvision 튜토리얼 기본 학습률


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

# Detection은 batch마다 box 개수가 달라서 collate_fn 필요
def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, collate_fn=collate_fn
)

# 4. 모델 준비
model = get_faster_rcnn(num_classes=num_classes)
model.to(device)

# 5. 옵티마이저
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

# 6. 학습 루프
# num_epochs 3번 나눠서 진행 
for epoch in range(3):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):   # batch_idx 추가
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

        # 중간 로그 (예: 1000 step마다 출력)
        if batch_idx % 1000 == 0:
            print(f"Epoch {epoch+1}, Step {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")


    # epoch마다 저장 (중복 제거)
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss,
    }, f"checkpoint_epoch{epoch+1}.pth")

print("첫 3 epoch 학습 완료, 체크포인트 저장됨")


