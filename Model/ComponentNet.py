import torch
import torchvision
from torch import nn
from torch.distributions import Normal

# class encoder(nn.Module):
#     def __init__(self):
#         super(encoder, self).__init__()
#         self.contract_layers = nn.Sequential(
#             nn.Conv2d(4, 32, 3, stride=2, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, stride=2, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, 3, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#
#         self.linear1 = nn.Linear(64 * 7 * 7, 256)
#         self.linear2_logvar = nn.Linear(256, 16)
#         self.linea2_mu = nn.Linear(256, 16)
#         self.relu = nn.ReLU()
#
#     def reparameterize(self, logvar, mu):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, image, mask):
#         x = torch.cat((image, mask), 1)
#         x = self.contract_layers(x)
#         bs, _, _, _ = x.shape
#         x = x.view(bs, 64 * 7 * 7)
#         x = self.linear1(x)
#         logvar = self.linear2_logvar(x)
#         mu = self.linea2_mu(x)
#         z = self.reparameterize(logvar, mu)
#         return z, logvar, mu
#
# # four convolutinal layer (conv1 - 4) to upsample the latent variable to size 128 * 128 * 32 and
# # two convolutional layer (conv5_mask, conv5_img) to output the logit(mask) (1 channel) and image (3 channel)
# # Get the mask by applying sigmoid on logit(mask)
# class decoder(nn.Module):
#     def __init__(self, inchannel):
#         super(decoder, self).__init__()
#         self.im_size = 128
#         self.conv1 = nn.Conv2d(inchannel + 2, 32, 3, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.conv4 = nn.Conv2d(32, 32, 3, bias=False)
#         self.bn4 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU()
#         self.inchannel = inchannel
#         self.conv5_img = nn.Conv2d(32, 3, 1)
#         self.conv5_mask = nn.Conv2d(32, 1, 1)
#
#         x = torch.linspace(-1, 1, self.im_size + 8)
#         y = torch.linspace(-1, 1, self.im_size + 8)
#         x_grid, y_grid = torch.meshgrid(x, y)
#         # Add as constant, with extra dims for N and C
#         self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
#         self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
#
#     def forward(self, z):
#         # z (bs, 32)
#         bs, _ = z.shape
#         z = z.view(z.shape + (1, 1))
#
#         # Tile across to match image size
#         # Shape: NxDx64x64
#         z = z.expand(-1, -1, self.im_size + 8, self.im_size + 8)
#
#         # Expand grids to batches and concatenate on the channel dimension
#         # Shape: Nx(D+2)x64x64
#         x = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
#                        self.y_grid.expand(bs, -1, -1, -1), z), dim=1)
#         # x (bs, 32, image_h, image_w)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
#         img = self.conv5_img(x)
#         logitmask = self.conv5_mask(x)
#         img = .5 + 0.55 * torch.tanh(img)
#
#         return img, logitmask
#
# class component_net(nn.Module):
#     def __init__(self):
#         super(component_net, self).__init__()
#         self.encoder = encoder()
#         self.decoder = decoder(16)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, mask):
#         z, logvar, mu = self.encoder(x, mask)
#         img, mask = self.decoder(z)
#         return logvar, mu, img, mask

class encoder(nn.Module):
    def __init__(self, z_size=16):
        super(encoder, self).__init__()
        self.contract_layers = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, bias=False),
            nn.CELU(),
            # nn.BatchNorm2d(64)
        )

        self.linear1 = nn.Linear(64 * 7 * 7, 256)
        self.linear2_logvar = nn.Linear(256, z_size)
        self.linea2_mu = nn.Linear(256, z_size)
        self.relu = nn.ReLU()

    def reparameterize(self, logvar, mu):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, mask):
        x = torch.cat((image, mask), 1)
        x = self.contract_layers(x)
        bs, _, _, _ = x.shape
        x = x.view(bs, -1)
        x = self.linear1(x)
        x = self.relu(x)
        logvar = self.linear2_logvar(x)
        mu = self.linea2_mu(x)
        # z = self.reparameterize(logvar, mu)
        m = Normal(mu, logvar)
        z = m.rsample()
        # z = mu
        return z, logvar, mu

# four convolutinal layer (conv1 - 4) to upsample the latent variable to size 128 * 128 * 32 and
# two convolutional layer (conv5_mask, conv5_img) to output the logit(mask) (1 channel) and image (3 channel)
# Get the mask by applying sigmoid on logit(mask)
class decoder(nn.Module):
    def __init__(self, inchannel):
        super(decoder, self).__init__()
        self.im_size = 128
        self.conv1 = nn.Conv2d(inchannel + 2, 32, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn4 = nn.BatchNorm2d(32)
        self.celu = nn.CELU()
        self.inchannel = inchannel
        self.conv5_img = nn.Conv2d(32, 3, 1)
        self.conv5_mask = nn.Conv2d(32, 1, 1)

        x = torch.linspace(-1, 1, self.im_size + 8)
        y = torch.linspace(-1, 1, self.im_size + 8)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
        self.bias = 0

    def forward(self, z):
        # z (bs, 32)
        bs, _ = z.shape
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size + 8, self.im_size + 8)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
                       self.y_grid.expand(bs, -1, -1, -1), z), dim=1)
        # x (bs, 32, image_h, image_w)
        x = self.conv1(x)
        x = self.celu(x)
        # x = self.bn1(x)
        x = self.conv2(x)
        x = self.celu(x)
        # x = self.bn2(x)
        x = self.conv3(x)
        x = self.celu(x)
        # x = self.bn3(x)
        x = self.conv4(x)
        x = self.celu(x)
        # x = self.bn4(x)
        img = self.conv5_img(x)
        img = .5 + 0.55 * torch.tanh(img + self.bias)
        logitmask = self.conv5_mask(x)

        return img, logitmask

# class decoder(nn.Module):
#     def __init__(self, inchannel):
#         super(decoder, self).__init__()
#         self.im_size = 128
#         expand_layer = []
#         expand_layer.append(nn.ConvTranspose2d(1, 32, 3, 2, 1, 1, bias=False))
#         expand_layer.append(nn.BatchNorm2d(32))
#         expand_layer.append(nn.ReLU())
#         for i in range(3):
#             expand_layer.append(nn.ConvTranspose2d(32, 32, 3, 2, 1, 1, bias=False))
#             nn.BatchNorm2d(32)
#             expand_layer.append(nn.ReLU())
#         self.expand_layer = nn.Sequential(*expand_layer)
#         self.conv_img = nn.Conv2d(32, 3, 3, padding=1)
#         self.conv_mask = nn.Conv2d(32, 1, 3, padding=1)
#
#     def forward(self, z):
#         # z (bs, 32)
#         z = z.reshape((-1, 1, 8, 8))
#         x = self.expand_layer(z)
#         # x = self.bn4(x)
#         img = self.conv_img(x)
#         img = torch.tanh(img)
#         logitmask = self.conv_mask(x)
#
#         return img, logitmask

class component_net(nn.Module):
    def __init__(self):
        super(component_net, self).__init__()
        self.encoder = encoder(64)
        self.decoder = decoder(64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        z, logvar, mu = self.encoder(x, mask)
        img, mask = self.decoder(z)
        return logvar, mu, img, mask