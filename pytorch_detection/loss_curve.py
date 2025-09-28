import torch
import matplotlib.pyplot as plt

losses = []
for e in range(1, 11):  # 1~6 epoch
    ckpt = torch.load(f"checkpoint/checkpoint_epoch{e}.pth")
    losses.append(ckpt["loss"])

plt.plot(range(1, len(losses)+1), losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Detection Training Loss")
plt.show()
