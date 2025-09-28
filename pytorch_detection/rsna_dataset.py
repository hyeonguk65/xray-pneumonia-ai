# DICOM 이미지를 불러와서 → float32 정규화 → RGB 변환
# CSV에서 바운딩박스를 읽어서 → [xmin, ymin, xmax, ymax]로 변환
# (이미지 Tensor, target 딕셔너리)를 리턴

import os
import torch
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class RSNADataset(Dataset):
    def __init__(self , img_dir , csv_file , transforms=None , resize=512):
        self.img_dir = img_dir
        self.labels = pd.read_csv(csv_file)
        self.transforms = transforms
        self.resize = resize 
        
        # NaN(비어있음) -> 0으로 채움 (쓸모없는 좌표 칸을 그냥 숫자 0) 
        self.labels = self.labels.fillna(0)
        
    # csv 행 개수 리턴
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_id = row["patientId"]
        img_path = os.path.join(self.img_dir, img_id + ".dcm")

        # 1. DICOM 읽기
        # DICOM은 보통 12~16비트 데이터 
        dcm = pydicom.dcmread(img_path)
        img = dcm.pixel_array.astype(np.float32)
        # img = dcm.pixel_array.astype(np.uint8) -> np.uint8은 0~255
        
        # 2. intensity 정규화 (0~1)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        #img = Image.fromarray(img).convert("RGB")
        
        # 3. H×W → PIL 변환 (RGB 3채널로 확장, detection 모델 입력 통일)
        img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        
        # 원래 이미지 크기
        orig_w, orig_h = img.size  

        # 4. 리사이즈
        img = img.resize((self.resize, self.resize))
        new_w, new_h = img.size

        # 5. 바운딩박스 & 라벨
        boxes, labels = [], []
        if row["Target"] == 1:
            x, y, w, h = row[["x", "y", "width", "height"]]

            # 좌표 스케일링
            x_scale = new_w / orig_w
            y_scale = new_h / orig_h

            xmin = x * x_scale
            ymin = y * y_scale
            xmax = (x + w) * x_scale
            ymax = (y + h) * y_scale

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # 폐렴

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        # 6. torchvision 변환 (Tensor 변환 등)
        if self.transforms:
            img = self.transforms(img)

        return img, target
    
    
