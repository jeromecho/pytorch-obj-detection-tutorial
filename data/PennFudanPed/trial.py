from dataset import PennFudanDataset
from aug import get_transform
import torch
import torchvision
import utils

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = PennFudanDataset('data/', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
  dataset,
  batch_size=2,
  shuffle=True,
  # setting 'num_workers' to a positive 
  #   integer ensures that our data loading
  #   does not block the remainder of 
  #   our code from computing
  num_workers=0,
  # collate function is the function 
  # used to collect data into mini batches
  collate_fn=utils.collate_fn
)

images, targets = next(iter(data_loader))
# Why not just list(images)?
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)
print(output)

model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)
print(predictions[0])


