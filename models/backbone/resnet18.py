import math
import torch
import re
import torch.nn as nn
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from batchnorm import SynchronizedBatchNorm2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=False, checkpoint_path=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        # self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained and checkpoint_path:
            self._load_pretrained_model(checkpoint_path)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_dict = {}
        state_dict = self.state_dict()

        # Remove specified layers ('classifier' and 'projector')
        checkpoint_dict = checkpoint['state_dict']
        for k, v in checkpoint_dict.items():
            if k.startswith('backbone.'):
                k = k[len('backbone.'):]
            if 'classifier' not in k and 'projector' not in k:
                if k in state_dict:
                    model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet18(output_stride, BatchNorm, checkpoint_path=None):
    """Constructs a ResNet-18 model.
    Args:
        checkpoint_path (str): Path to custom checkpoint file
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, pretrained=False, checkpoint_path=checkpoint_path)
    # model = nn.Sequential(*list(model.children())[:-1]) 
    return model

def check_checkpoint_structure(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = model.state_dict()

    print(f"Model's state_dict keys:")
    for key in model_state_dict.keys():
        print(key)

    print(f"\nCheckpoint's state_dict keys:")
    for key in checkpoint['state_dict'].keys():
        if key.startswith('backbone.'):
            new_key = key[len('backbone.'):]  # Remove 'backbone.' prefix
        else:
            new_key = key
        
        if 'classifier' not in new_key and 'projector' not in new_key:
            if new_key in model_state_dict:
                print(f"Key '{new_key}' found in model's state_dict!")
            else:
                print(f"Key '{new_key}' not found in model's state_dict!")

if __name__ == "__main__":
    import torch
    # Replace with your actual checkpoint path
    checkpoint_path = "/PATH/TO/SIMCLR/checkpoint.ckpt"
    model = ResNet18(BatchNorm=nn.BatchNorm2d, checkpoint_path=checkpoint_path, output_stride=8)
    check_checkpoint_structure(checkpoint_path, model)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
