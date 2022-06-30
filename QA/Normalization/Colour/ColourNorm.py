import torch.nn as nn
import torch
from typing import Union
import numpy as np
from matplotlib import pyplot as plt


# Collection of classes that are instances of nn.Module meant to be used in scriptable
# transforms (https://pytorch.org/vision/stable/transforms.html#scriptable-transforms).
# The forward() method of each class is used as the transform.

class Macenko(nn.Module):
    # Macenko colour normalisation. Takes as an input a torch tensor that goes from 0 to 255
    # of size (C, H, W) and outputs a colour-normalised array of the same size (C, H, W).
    # Inspired by: https://github.com/EIDOSlab/torchstain/blob/main/torchstain/normalizers/torch_macenko_normalizer.py

    # INPUTS
    # alpha: percentile for normalisation (considers data within alpha and (100-alpha) percentiles).
    # beta: threshold of normalisation values for analysis.
    # saved_fit_file: (optional) path of tensor with pre-trained parameters HERef and maxCRef.

    def __init__(self, alpha=1, beta=0.15, Io=240, saved_fit_file=None, get_stains=False):
        super(Macenko, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.get_stains = get_stains
        self.Io = Io
        # Default fit values reported in the original git code (origin unclear)
        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])

        if saved_fit_file is not None:  # then you have a pre-fitted dataset!
            temp = torch.load(saved_fit_file)
            self.alpha = temp['alpha']
            self.beta = temp['beta']
            self.Io = temp['Io']
            self.HERef = temp['HERef']
            self.maxCRef = temp['maxCRef']

    def forward(self, img, fit=None):
        # img should be of size C x H x W.

        if fit is not None:
            self.fit(img)

        img_norm, H, E = self.normalize(torch.mul(img, 255.0), stains=self.get_stains)
        img_norm = torch.div(img_norm, 255.0)

        # Following normalization, img_norm will be of shape H x W x C -> return as C x H x W!
        if self.get_stains:
            return img_norm.permute(2, 0, 1), H, E
        else:
            return img_norm.permute(2, 0, 1)

    def convert_rgb2od(self, img):
        img = img.permute(1, 2, 0)
        OD = -torch.log((img.reshape((-1, img.shape[-1])).float() + 1) / self.Io)
        ODhat = OD[~torch.any(OD < self.beta, dim=1)]  # remove transparent pixels

        return OD, ODhat

    def find_HE(self, ODhat, eigvecs):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = percentile(phi, self.alpha)
        maxPhi = percentile(phi, 100 - self.alpha)

        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        return HE

    def find_concentration(self, OD, HE):

        Y = OD.T  # rows correspond to channels (RGB), columns to OD values
        out = torch.linalg.lstsq(HE, Y)[0]  # determine concentrations of the individual stains

        return out

    def compute_matrices(self, img):
        OD, ODhat = self.convert_rgb2od(img)

        if ODhat.shape[0] <= 10:  # this slide is bad for processing - too many transparent points.
            HE = None
            C = None
            maxC = None
        else:
            _, eigvecs = torch.linalg.eigh(cov(ODhat.T), UPLO='U')  # or L?
            eigvecs = eigvecs[:, [1, 2]]
            HE = self.find_HE(ODhat, eigvecs)
            C = self.find_concentration(OD, HE)
            maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, img):
        HE, _, maxC = self.compute_matrices(img)
        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, img, stains=True):
        # Input:
        # img: tensor of size C x H x W, 0 to 255 intensity
        # Io: (optional) transmitted light intensity
        # alpha: percentile
        # beta: transparency threshold
        # stains: if true, returns H&E components

        #Output:
        # I_norm: colour normalised image, 0 to 255 intensity
        # H: hematoxylin image
        # E: eosin image

        c, h, w = img.shape
        H, E = None, None
        HE, C, maxC = self.compute_matrices(img)

        if HE is not None:

            C *= (self.maxCRef / maxC).unsqueeze(-1)  # normalize stain concentrations

            # recreate the image using reference mixing matrix
            Inorm = self.Io * torch.exp(-torch.matmul(self.HERef, C))
            Inorm[Inorm > 255] = 255
            Inorm = Inorm.T.reshape(h, w, c).int()

            if stains:
                H = torch.mul(self.Io, torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
                H[H > 255] = 255
                H = H.T.reshape(h, w, c).int()

                E = torch.mul(self.Io, torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
                E[E > 255] = 255
                E = E.T.reshape(h, w, c).int()

        else:  # then this means the current patch is not good for processing - conserve the same image.
            Inorm = img.reshape(h, w, c).int()

        return Inorm, H, E


def cov(x):
    E_x = x.mean(dim=1)
    x = x - E_x[:, None]
    return torch.mm(x, x.T) / (x.size(1) - 1)


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    k = 1 + round(.01 * float(q) * (t.numel() - 1))

    if k == 0 or t.size()[0] == 0:
        out = torch.tensor(0)  # default to dummy value if there is no point of interest in slide. Will not affect
        # the results, this is just a fail safe.
    else:
        out = t.view(-1).kthvalue(int(k)).values

    return out
