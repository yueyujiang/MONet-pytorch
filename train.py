import argparse
import os

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn

from tensorboardX import SummaryWriter
import Model.Monet as Monet
from data import MultiDSprites
import torch.optim as optim
import utils
import shutil
import glob

parser = argparse.ArgumentParser(description='Test!')
parser.add_argument('-s', '--slot', type=int, default=4,
                    help="number of slots in attention net")
parser.add_argument('-c', '--checkpoint-save-pth', type=str, default='./checkpoint',
                    help="path for saving the model\'s parameters")
parser.add_argument('-l', '--load-parameters', type=str, default=None,
                    help='path for parameters file')
parser.add_argument('-e', '--epochs', type=int, default=8000,
                    help="number of epochs to train the model")
parser.add_argument('-f', '--frequency', type=int, default=50,
                    help="frequency for ploting the figure and saving the values of parameters")
parser.add_argument('--lr', type=float, default=1e-4,
                    help="learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=3,
                    help="batch size")
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Monet.monet(args.slot)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

checkpoint_path = args.checkpoint_save_pth
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

if args.load_parameters:
    utils.load_checkpoint(model, args.load_parameters, optimizer)

transform = transforms.Compose([transforms.CenterCrop((320, 320)),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor()])

trainset = MultiDSprites(root='data', train=True)

trainloader = DataLoader(trainset, batch_size=args.batch_size,
                         shuffle=True, num_workers=8)
BCEloss = nn.BCELoss().to(device)

writer = SummaryWriter(args.summary_dir)
for epoch in range(args.epochs):  # loop over the dataset multiple times
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    running_loss3 = 0
    for batch_idx, sample in enumerate(trainloader):
        sample = sample.to(device)
        reconstruction_image, logmasks, history_logsk, recon_img, loss1, loss2, loss3, loss = model(sample)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.step()
        running_loss += loss
        running_loss1 += loss1
        running_loss2 += loss2
        running_loss3 += loss3
        # Below is some codes for plotting the data
        if batch_idx % args.frequency == args.frequency - 1:
            grid_image = make_grid(reconstruction_image, 6, normalize=True, pad_value=1)
            writer.add_image('reconstruction_image', grid_image, batch_idx)
            grid_image = make_grid(sample, 6, normalize=True, pad_value=1)
            writer.add_image('original image', grid_image, batch_idx)
            for i in range(args.batch_size):
                grid_image = make_grid(torch.exp(logmasks[i, :, :, :]).unsqueeze(1), 6, normalize=True, pad_value=1)
                writer.add_image('mask' + str(i), grid_image, batch_idx)
                grid_image = make_grid(torch.exp(history_logsk[i, :, :, :]).unsqueeze(1), 6, normalize=True, pad_value=1)
                writer.add_image('sk' + str(i), grid_image, batch_idx)
                grid_image = make_grid(recon_img[i, :, :, :, :].permute((3, 0, 1, 2)), 6, normalize=True, pad_value=1)
                writer.add_image('recon_img' + str(i), grid_image, batch_idx)
            print("epoch", epoch, "batch", batch_idx)
            print('running loss', running_loss.item())
            writer.add_scalar('loss', running_loss, (epoch) * len(trainloader) + batch_idx)
            writer.add_scalar('loss1', running_loss1, (epoch) * len(trainloader) + batch_idx)
            writer.add_scalar('loss2', running_loss2, (epoch) * len(trainloader) + batch_idx)
            writer.add_scalar('loss3', running_loss3, (epoch) * len(trainloader) + batch_idx)
            running_loss = 0
            running_loss1 = 0
            running_loss2 = 0
            running_loss3 = 0
            utils.save_checkpoint(model, epoch, batch_idx, optimizer, checkpoint_PATH=checkpoint_path)
    list_of_files = glob.glob('checkpoint/*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_file = latest_file.split('/')[1]
    shutil.move(os.path.join('checkpoint', latest_file), os.path.join('backup', latest_file))
    shutil.rmtree('checkpoint')
    os.mkdir(args.checkpoint_save_pth)
