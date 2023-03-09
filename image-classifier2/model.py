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

    def __init__(self, num_classes=2):
        super(ImageClassifierNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

