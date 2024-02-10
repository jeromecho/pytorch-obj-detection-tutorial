import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Note: Again, "fpn" -  Feature Pyramid Network?
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 2 # (person, BG)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Q: is the 'head' of a neural network its input layer?
#    What does the architecture of this look like?
# Original RCNN paper: https://arxiv.org/pdf/1311.2524v5.pdf    
# What is a 'predictor' more generally speaking?
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

