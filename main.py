import torch
import torchvision
import torchvision.transforms as transforms

from modules.training.trainer import Trainer
from modules.models.model import Model
from modules.models.simple_model import SimpleModel


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


model = SimpleModel()

if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
        print("GPU available.")
else:
    cuda_device = -1

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

trainer = Trainer(model, optimizer, train_dataloader=trainloader, validation_dataloader=testloader, cuda_device=cuda_device)
trainer.train()
    