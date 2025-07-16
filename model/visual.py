from torchviz import make_dot
import torch

from model.FPN import FPN ,num_classes
# Generate graph
model = FPN(num_classes=num_classes)
x = torch.randn(1, 3, 800, 1200)
y = model(x)

# Create computational graph
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("fpn_architecture", format="png")