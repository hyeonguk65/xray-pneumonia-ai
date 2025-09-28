# torchvision에서 제공하는 Faster R-CNN
# ResNet50 + FPN

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn(num_classes):
    # 미리 학습된 Faster R-CNN 모델 불러오기 (ResNet50 + FPN 백본)
    # weights="DEFAULT" → ImageNet + COCO 데이터로 사전학습된 가중치 로드
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # 출력층을 우리가 원하는 클래스 수에 맞게 교체
    # (배경 0 +폐렴1 -> 총 2개 클래스)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


