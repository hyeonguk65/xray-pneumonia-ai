# 데이터 불러오기 
# 이미지를 “모델이 이해할 수 있는 숫자 데이터”로 변환
# 전처리 정의 → 데이터셋 로드 → 데이터로더 생성 → 클래스 확인

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(data_dir=r"C:\Users\user\Desktop\project\chest_xray" , batch_size = 32):
    
    # 데이터 전처리 파이프라인 정의
    data_transforms = {
        "train": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),   # 흑백 이미지를 3채널로 확장 (사전학습 모델 호환)
            transforms.Resize(256, antialias=True),        # 짧은 변 기준 256으로 리사이즈 (비율 유지)
            transforms.CenterCrop(224),                    # 중앙에서 224x224 크기로 자르기 (ResNet 표준 입력 크기)
            transforms.RandomRotation(degrees=7, fill=0),  # 데이터 증강: 약간의 회전 (±7°)
            transforms.ToTensor(),                         # 이미지를 PyTorch Tensor로 변환
            transforms.Normalize([0.485,0.456,0.406],      # 평균값으로 정규화
                                [0.229,0.224,0.225]),     # 표준편차로 정규화 (ImageNet 기준)
        ]),

        "val": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),

        "test": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
    }
    
    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(
        r"C:\Users\user\Desktop\project\chest_xray\train",  
        transform=data_transforms["train"]
    )

    val_dataset = datasets.ImageFolder(
        r"C:\Users\user\Desktop\project\chest_xray\val",
        transform=data_transforms["val"]
    )

    test_dataset = datasets.ImageFolder(
        r"C:\Users\user\Desktop\project\chest_xray\test",
        transform=data_transforms["test"]
    )
    
    # 데이터 로더 정의
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   # 학습 데이터는 셔플
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)   # 검증/테스트는 순서 유지
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 클래스 확인
    class_names = train_dataset.classes
    print("클래스:", class_names)  # ['NORMAL', 'PNEUMONIA']
    
    # 데이터셋 크기 (정확도 계산용)
    # dataset_sizes는 정확도 계산과 훈련 로그 출력을 깔끔하게 하기 위해 필요
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }

    return train_loader, val_loader, test_loader, dataset_sizes, class_names
    

