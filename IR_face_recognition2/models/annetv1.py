import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AnNet', 'annet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AnNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AnNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3), # 56*56*16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1), # 28*28*16

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),# 14*14*32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),# 7*7*32

            nn.Conv2d(32, 64, kernel_size=3, stride=2), # 3*3*64
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 3 * 3, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 3 * 3)
        x = self.classifier(x)
        return x

def annet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AnNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['annet']))
    return model
