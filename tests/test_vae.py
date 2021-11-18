import torch
from torchsummary import summary
from models import VanillaVAE

def test_forward():
    model = VanillaVAE(784, 400, 20)
    x = torch.randn(1, 784)
    y = model(x)
    print("Model Output size:", y[0].size())

def test_loss():
    model = VanillaVAE(784, 400, 20)
    x = torch.randn(1, 784)
    result = model(x)
    loss = model.loss_function(*result)
    print(loss)