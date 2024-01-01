# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch
import torch.optim as optim

import numpy as np

from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake

def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "ucat":
        kwargs.setdefault('epoch', 105)
        kwargs.setdefault('batch_size', 128)
        patch_size = kwargs.setdefault('patch_size', 24)
        center_pixel = False
        model = ucat(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.03)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('scheduler', optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=4))
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBnRelu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

class ConvBn(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBn, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

class SpeGSA(nn.Module):
    def __init__(self, width1, width2):
        super(SpeGSA, self).__init__()
        self.groups = width1//3

        self.pool1 = nn.MaxPool2d(4,4)
        self.pool2 = nn.AvgPool2d(4,4)

        self.attnconv = nn.Conv2d(self.groups, self.groups, kernel_size=(3,1), stride=(3,1), groups=self.groups, bias=False)

        self.bn = nn.BatchNorm2d(self.groups)
        self.relu = nn.ReLU(inplace=True)

        self.conv = ConvBn(self.groups, width2)

    def forward(self, x):
        b, c, h, w = x.shape

        xq = self.pool1(x)
        xq = xq.view(b, self.groups, c // self.groups, h*w//16)

        xk = self.pool2(x)
        xk = xk.view(b, self.groups, c // self.groups, h * w // 16).transpose(-1, -2)

        xv = x.view(b, self.groups, c // self.groups, h*w)

        attn = ((xq @ xk) * (4/h))
        attn = self.attnconv(attn)
        attn = attn.softmax(dim=-1)
        out = attn @ xv
        out = out.reshape(b,self.groups,h,w)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv(out)
        return out

class SpaDCSA(nn.Module):
    def __init__(self, width, stride, groups):
        super(SpaDCSA, self).__init__()
        self.groups = groups

        self.stride = stride
        if self.stride == 2:
            self.convq = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=2, stride=2)
            self.convk = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                   kernel_size=2, stride=2)
            self.convv = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                   kernel_size=2, stride=2)
        else:
            self.convq = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                   kernel_size=3, stride=1, padding=1)
            self.convk = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                   kernel_size=1, stride=1, padding=0)
            self.convv = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                   kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xq = self.convq(x)
        b, c, h, w = xq.shape
        xq = xq.view(b, self.groups, c // self.groups, h, w).flatten(3).transpose(-1, -2)

        xk = self.convk(x)
        xk = xk.view(b, self.groups, c // self.groups, h, w).flatten(3)

        xv = self.convv(x)
        xv = xv.view(b, self.groups, c // self.groups, h, w).flatten(3).transpose(-1, -2)

        attn = (xq @ xk) * ((c // self.groups) ** -0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ xv
        out = out.transpose(-1, -2).reshape(b,c,h,w)
        out = self.bn(out)
        out = self.relu(out)
        return out

class SpaCCA(nn.Module):
    def __init__(self, width, stride, groups):
        super(SpaCCA, self).__init__()
        self.groups = groups

        self.stride = stride
        if self.stride == 2:
            self.convq = nn.ConvTranspose2d(in_channels=width, out_channels=width, groups=groups,
                                         kernel_size=2, stride=2)
        else:
            self.convq = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                kernel_size=1, stride=1)

        self.convk = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                kernel_size=1, stride=1)

        self.convv = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                kernel_size=1, stride=1)

        self.bn = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        xq = self.convq(x2)
        b, c, h, w = xq.shape
        xq = xq.view(b, self.groups, c // self.groups, h, w).flatten(3).transpose(-1, -2)

        xk = self.convk(x1)
        xk = xk.view(b, self.groups, c // self.groups, h, w).flatten(3)

        xv = self.convv(x1)
        xv = xv.view(b, self.groups, c // self.groups, h, w).flatten(3).transpose(-1, -2)

        attn = (xq @ xk) * ((c // self.groups) ** -0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ xv
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Spe(nn.Module):
    def __init__(self, channel1, channel2):
        super(Spe, self).__init__()
        self.width = (channel1//3)*3
        self.attn = SpeGSA(self.width, channel2)
        self.idn = ConvBn(channel1, channel2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        idn = self.idn(x)
        x = x[:,:self.width,:,:]
        out = self.attn(x)
        out = idn + out
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, channel, stride, groups):
        super(Encoder, self).__init__()
        self.conv1 = ConvBnRelu(channel, channel)
        self.attn = SpaDCSA(channel, stride, groups)
        self.conv2 = ConvBn(channel, channel)
        self.stride = stride
        if self.stride == 2:
            self.idn = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(channel))
        else:
            self.idn = nn.Identity()
        self.drop_path = DropPath(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        idn = self.idn(x)
        out = self.conv1(x)
        out = self.attn(out)
        out = self.conv2(out)
        out = idn + self.drop_path(out)
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, channel, stride, groups):
        super(Decoder, self).__init__()
        self.conv1 = ConvBnRelu(channel, channel)
        self.attn = SpaCCA(channel, stride, groups)
        self.conv2 = ConvBn(channel, channel)
        self.stride = stride
        if self.stride == 2:
            self.idn = nn.Sequential(
                nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(channel))
        else:
            self.idn = nn.Identity()
        self.drop_path = DropPath(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        idn = self.idn(x2)
        out = self.conv1(x2)
        out = self.attn(x1, out)
        out = self.conv2(out)
        out = idn + self.drop_path(out)
        out = self.relu(out)
        return out

class ucat(nn.Module):
    def __init__(self, n_bands, n_classes):
        super(ucat, self).__init__()
        self.in_channel = 64
        self.g = 8

        self.layer0 = Spe(n_bands, self.in_channel)

        self.layer1 = Encoder(self.in_channel, 2, self.g)
        self.layer2 = Encoder(self.in_channel, 1, self.g)
        self.layer3 = Encoder(self.in_channel, 2, self.g)
        self.layer4 = Encoder(self.in_channel, 1, self.g)
        self.layer5 = Encoder(self.in_channel, 1, self.g)
        self.layer6 = Decoder(self.in_channel, 1, self.g)
        self.layer7 = Decoder(self.in_channel, 1, self.g)
        self.layer8 = Decoder(self.in_channel, 2, self.g)
        self.layer9 = Decoder(self.in_channel, 1, self.g)

        self.up = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=n_classes,
                                     kernel_size=2, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.squeeze(1)
        x0 = self.layer0(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x4, x5)
        x7 = self.layer7(x3, x6)
        x8 = self.layer8(x2, x7)
        x9 = self.layer9(x1, x8)
        x10 = self.up(x9)
        return x10

def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    device=torch.device("cpu"),
    supervision="full",
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    for e in tqdm(range(1, epoch), desc="Training the network"):
        # Set the network to training mode
        net.train()

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == "full":
                output = net(data)
                loss = criterion(output, target)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            del (data, target, loss, output)

        if scheduler is not None:
            scheduler.step()

def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs

