import torch.optim
from torch import nn

import Model.AttentionNet as AttentionNet
import Model.ComponentNet as ComponentNet
import utils

class monet(nn.Module):
    def __init__(self, slot, train=True):
        super(monet, self).__init__()
        self.atten_net = AttentionNet.attention_net(slots=slot)
        self.comp_net = ComponentNet.component_net()
        self.train = train

    def forward(self, sample):
        logmasks, history_logsk = self.atten_net(sample)
        reconstruction_image = torch.zeros(sample.shape).to(sample.device)
        if self.train:
            loss1 = 0
            loss2 = 0
        for i in range(logmasks.shape[1]):
            logvar, mu, recon_img, logitrecon_mask = self.comp_net(sample, logmasks[:, i, :, :].unsqueeze(1))
            mask = torch.exp(logmasks[:, i, :, :].unsqueeze(1))
            if self.train:
                loss1 += torch.sum((mask * sample - mask * recon_img) ** 2 / 0.0225)
                loss2 += utils.normal_KL_div_loss(logvar, mu)
            if i == 0:
                loss3_l = logitrecon_mask
                recon_images = recon_img.unsqueeze(4)
            else:
                loss3_l = torch.cat((loss3_l, logitrecon_mask), 1)
                recon_images = torch.cat((recon_images, recon_img.unsqueeze(4)), 4)
            reconstruction_image += recon_img * mask
        logrecon_masks = torch.nn.functional.log_softmax(loss3_l, dim=1)
        if self.train:
            loss3 = utils.cross_entropy(logmasks, logrecon_masks)
            loss = loss1 + 0.25 * loss2 + 0.025 * loss3
        else:
            loss = loss1 = loss2 = loss3 = 0
        if not self.train:
            return reconstruction_image.detach(), logmasks.detach(), history_logsk.detach(), recon_images.detach()
        return reconstruction_image, logmasks, history_logsk, recon_images, loss1, loss2, loss3, loss
