import torch
import torchvision
from torch import nn

class double_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(double_conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class unet_downsample_block(nn.Module):
    def __init__(self, inchannel, outchannel, downsample):
        super(unet_downsample_block, self).__init__()
        self.conv = double_conv(inchannel, outchannel)
        if downsample:
            self.output_layer = nn.MaxPool2d(2)
        self.downsample = downsample

    def forward(self, x):
        skip = self.conv(x)
        if self.downsample:
            y = self.output_layer(skip)
        else:
            y = skip
        return skip, y

class unet_upsample_block(nn.Module):
    def __init__(self, inchannel, outchannel, upsample):
        super(unet_upsample_block, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(inchannel * 2, inchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.upsample = upsample

    def forward(self, x, skip):
        x = torch.cat((skip, x), 1)
        x = self.double_conv(x)
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2)
        y = x
        return y


class attention_net(nn.Module):
    def __init__(self, slots):
        super(attention_net, self).__init__()
        # define contracting path
        self.down_block1 = unet_downsample_block(4, 64, True)
        self.down_block2 = unet_downsample_block(64, 128, True)
        self.down_block3 = unet_downsample_block(128, 256, True)
        self.down_block4 = unet_downsample_block(256, 512, True)
        # self.down_block5 = unet_downsample_block(512, 512, False)
        self.down_block5 = unet_downsample_block(512, 1024, True)
        self.down_block6 = unet_downsample_block(1024, 1024, False)

        self.linear1 = nn.Linear(4 * 4 * 1024, 128)
        # self.linear1 = nn.Linear(8 * 8 * 512, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 4 * 4 * 1024)

        # define expansive path
        self.up_block1 = unet_upsample_block(1024, 1024, True)
        self.up_block2 = unet_upsample_block(1024, 512, True)
        self.up_block3 = unet_upsample_block(512, 256, True)
        self.up_block4 = unet_upsample_block(256, 128, True)
        self.up_block5 = unet_upsample_block(128, 64, True)
        self.up_block6 = unet_upsample_block(64, 64, False)

        self.layer = nn.Conv2d(64, 2, 1)
        self.relu = nn.ReLU()
        self.slots = slots

    def forward(self, x):
        bs, channel0, w0, h0 = x.shape
        logsk = torch.zeros([bs, 1, w0, h0]).to(x.device).requires_grad_(False)
        history_logsk = logsk
        # Calculate log(mask)
        for i in range(self.slots):
            x_in = torch.cat((x, logsk), 1)
            skip1, x1 = self.down_block1(x_in)
            skip2, x2 = self.down_block2(x1)
            skip3, x3 = self.down_block3(x2)
            skip4, x4 = self.down_block4(x3)
            skip5, x5 = self.down_block5(x4)
            skip6, x6 = self.down_block6(x5)
            bs, channel1, w, h = x6.shape

            h0 = x6.view([bs, channel1 * w * h])
            h1 = self.linear1(h0)
            h2 = self.linear2(h1)
            h3 = self.linear3(h2)
            h3 = self.relu(h3)

            y0 = h3.view([bs, channel1, w, h])

            y1 = self.up_block1(y0, skip6)
            y2 = self.up_block2(y1, skip5)
            y3 = self.up_block3(y2, skip4)
            y4 = self.up_block4(y3, skip3)
            y5 = self.up_block5(y4, skip2)
            y6 = self.up_block6(y5, skip1)

            y = self.layer(y6)
            # y has 2 channel for alpha and 1-alpha respectively, use softmax to make use they sum up to one
            tmp = nn.functional.log_softmax(y, dim=1)
            logalpha = tmp[:, 1, :, :].unsqueeze(1)
            log1_alpha = tmp[:, 0, :, :].unsqueeze(1)
            if i == self.slots - 1:
                logmk = logsk
            else:
                logmk = logsk + logalpha
                logsk = logsk + log1_alpha
            if i == 0:
                ans = logmk
            else:
                ans = torch.cat((ans, logmk), 1)
            history_logsk = torch.cat((history_logsk, logsk), 1)
        return ans, history_logsk
