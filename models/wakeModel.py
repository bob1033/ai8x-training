import torch.nn as nn

import ai8x
  
class wakeModel_deep(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_channels=48,
        dimensions=(64,64),
        bias=True,
        **kwargs
    ):
        super().__init__()
        
        self.conv1_1 = ai8x.FusedConv2dReLU(num_channels, 32, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv1_2 = ai8x.FusedConv2dReLU(32, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv1_3 = ai8x.FusedConv2dReLU(64, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv1_4 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv1_5 = ai8x.FusedConv2dReLU(64, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv1_6 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        
        self.conv2_1 = ai8x.FusedMaxPoolConv2dReLU(64, 96, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2_2 = ai8x.FusedConv2dReLU(96, 96, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2_3 = ai8x.FusedMaxPoolConv2dReLU(96, 96, 3, stride=1, padding=1, bias=bias,  **kwargs)
        self.conv2_4 = ai8x.FusedConv2dReLU(96, 96, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2_5 = ai8x.FusedConv2dReLU(96, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2_6 = ai8x.FusedConv2dReLU(64, 96, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2_7 = ai8x.FusedConv2dReLU(96, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        
        #self.fc1 = ai8x.FusedLinearReLU(4*4*64,128, bias=bias)
        self.fc1 = ai8x.Linear(4*4*64, num_classes, wide=True, bias=True)
        
        self.dropout = nn.Dropout(.3)
        self.dropout_lite = nn.Dropout(.1)
        self.dropout_medium = nn.Dropout(.2)
        
    def forward(self,x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = self.conv1_5(x)
        x = self.conv1_6(x)
        x = self.conv2_1(x)
        x = self.dropout_lite(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        x = self.dropout_lite(x)
        x = self.conv2_5(x)
        x = self.dropout_medium(x)
        x = self.conv2_6(x)
        x = self.dropout_medium(x)
        x = self.conv2_7(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def wakemodel_deep(pretrained=False, **kwargs):
    assert not pretrained
    return wakeModel_deep(**kwargs)
