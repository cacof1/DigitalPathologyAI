import torch.nn as nn
import torch
from typing import Union
import numpy as np


class Macenko(nn.Module):
    # Macenko colour normalisation. Takes as input a torch tensor (intensity ranging [0,255])
    # alpha: percentile for normalisation (considers data within alpha and (100-alpha) percentiles).
    # beta: threshold of normalisation values for analysis.
    # Io: (optional) transmitted light intensity
    def __init__(self, alpha=1, beta=0.15, Io=255, get_stains=False, normalize=True, HERef=None, maxCRef=None):
        super(Macenko, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.get_stains = get_stains
        self.Io = Io
        self.normalize = normalize
        # Default fit values are from svs file ID 484813.

        if HERef is None:
            self.HERef = torch.tensor([[0.5571, 0.2586],
                                       [0.7529, 0.7411],
                                       [0.3503, 0.6196]])
        else:
            self.HERef = torch.tensor(HERef)

        if maxCRef is None:
            self.maxCRef = torch.tensor([1.2955, 0.8696])
        else:
            self.maxCRef = torch.tensor(maxCRef)

    def forward(self, img, HE=None, maxC=None):  # img has expected size C x H x W
        c, h, w = img.shape
        img = img.reshape(img.shape[0], -1)  # Collapse H x W -> N
        OD, ids = self.convert_rgb2od(img)

        if OD[:, ids].shape[1] <= 10:  # this slide is bad for processing - too many transparent points.
            img = img.reshape(c, h, w)
            if self.get_stains:
                return img, None, None, None, self.HERef
            else:
                return img

        C = torch.linalg.lstsq(HE, OD[:, ids])[0]  # determine concentrations of the individual stains
        if maxC is None:  # otherwise use the input.
            maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        if (self.normalize): C *= (self.maxCRef / maxC).unsqueeze(-1)  # normalize stain concentrations

        # recreate the image using reference stain vectors
        img[:, ids] = torch.exp(-torch.matmul(self.HERef, C))
        img = img.reshape(c, h, w)

        if self.get_stains:
            H = torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0)))
            H = H.T.reshape(c, h, w)

            E = torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0)))
            E = E.T.reshape(c, h, w)
            return img, H, E, self.HE, maxC
        else:
            return img

    def convert_rgb2od(self, img):  # img has expected size C x H x W
        OD = -torch.log(img)
        ids = torch.logical_not(torch.logical_or(torch.any(OD.isinf(), dim=0), torch.any(OD < self.beta, dim=0)))
        return OD, ids

    def find_HE(self, img, get_maxC=False):  # img has expected size C x H x W
        img = img.reshape(img.shape[0], -1)  # Collapse H x W -> N

        OD, ids = self.convert_rgb2od(img)
        if OD[:, ids].shape[1] <= 10:
            if (get_maxC):
                return self.HERef, self.maxCRef  # this slide is bad for processing - too many transparent points.
            else:
                return self.HERef

        eigvalues, eigvecs = torch.linalg.eigh(torch.cov(OD[:, ids]), UPLO='L')
        eigvecs = eigvecs[:, [1, 2]]

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        That = torch.matmul(OD[:, ids].T, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = percentile(phi, self.alpha)
        maxPhi = percentile(phi, 100 - self.alpha)
        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        if get_maxC:
            C = torch.linalg.lstsq(HE, OD[:, ids])[0]  # determine concentrations of the individual stains
            maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
            return HE, maxC
        else:
            return HE


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
