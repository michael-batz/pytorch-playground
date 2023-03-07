import torch
import torchvision


class ImageClassifierSettings:

    def __init__(self, model_file=None):
        # define the transformation of images
        self.image_transformer = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.458, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])

        # setup the device
        self.device = "cpu"

        # define the batch size
        self.batch_size = 64

        # define the epoch size
        self.epochs = 20

        # create the neuronal network
        if model_file:
            # load from file if defined
            self.model = torch.load(model_file)
        else:
            # if not, create a new one
            self.model = ImageClassifierNet()
        self.model.to(self.device)

        # set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # setup the loss function
        self.loss_function = torch.nn.CrossEntropyLoss()


class ImageClassifierNet(torch.nn.Module):

    def __init__(self):
        super(ImageClassifierNet, self).__init__()
        self.fc1 = torch.nn.Linear(12288, 84)
        self.fc2 = torch.nn.Linear(84, 50)
        self.fc3 = torch.nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

