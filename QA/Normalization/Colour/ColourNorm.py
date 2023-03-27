import torch.nn as nn
import torch
from typing import Union
import numpy as np

class Macenko(nn.Module):
    # Macenko colour normalisation.
    # Inspired by: https://github.com/EIDOSlab/torchstain/blob/main/torchstain/normalizers/torch_macenko_normalizer.py

    def __init__(self, alpha=1, beta=0.15, normalise_concentration=True, HE_jitter=False, saved_fit_file=None, get_stains=False):
        super(Macenko, self).__init__()
        self.Io = 255  # normalisation value (for RGB...)
        self.alpha = alpha  # percentile for normalisation (considers data within alpha and (100-alpha) percentiles).
        self.beta = beta  # transparency threshold
        self.get_stains = get_stains
        self.normalise_concentration = normalise_concentration  # bool to normalise stain concentrations with respect to maxCRef.
        self.HE_jitter = HE_jitter # tuple with (mean, std) to perturb H&E stain vectors. For colour augmentation.
        # Default fit values (from slide ID 484813)
        self.HERef = torch.tensor([[0.5571, 0.2586],
                                   [0.7529, 0.7411],
                                   [0.3503, 0.6196]])
        self.maxCRef = torch.tensor([1.2955, 0.8696])

        if saved_fit_file is not None:  # (optional) path of tensor with pre-trained parameters HERef and maxCRef.
            temp = torch.load(saved_fit_file)
            self.alpha = temp['alpha']
            self.beta = temp['beta']
            self.HERef = temp['HERef']
            self.maxCRef = temp['maxCRef']

    def forward(self, img, fit=False, HE_test=None, maxC_test=None):
        # img: float32 torch tensor(intensity ranging[0, 1]) of size (c, h, w)
        # fit: bool to specify if HERef, maxCRef should be fitted to the current dataset.
        # HE_test: pre-calculated stain vectors (if you do not want to infer on tile, and maybe use WSI-wise)
        # maxC_test: pre-calculated max concentration (if you do not want to infer on tile, and maybe use WSI-wise)

        # img_norm: colour-normalised float32 torch tensor of the same size and range.
        # H, E: float32 torch tensors of size (c, h, w) representing stain concentrations.

        c, h, w = img.shape
        img = img.reshape(img.shape[0], -1)  # collapse (C, H, W) to (C, H*W)

        # Calculate self.HEref, self.maxCRef on current dataset if not computed yet.
        if fit:
            self.fit(img)

        #########################################################################################################################

        HE_test, C, maxC_test = self.compute_matrices(img, HE_test=HE_test, maxC_test=maxC_test)


        if (HE_test is None) or (C is None):
            return img.reshape(c, h, w)  # then the image was too transparent for Macenko, return the same.
        else:
            if self.normalise_concentration:
                C *= (self.maxCRef / maxC_test).unsqueeze(-1)  # normalize stain concentrations

            # recreate the image using reference mixing matrix
            if self.HE_jitter:
                HE_perturbed = torch.mul(self.HERef, self.HE_jitter[0] + torch.mul(self.HE_jitter[1], torch.randn(2)))
                img_norm = torch.exp(-torch.matmul(HE_perturbed, C))
            else:
                img_norm = torch.exp(-torch.matmul(self.HERef, C))
            img_norm[img_norm > 1.0] = 1.0

            if self.get_stains:

                H = torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0)))
                H[H > 1.0] = 1.0

                E = torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0)))
                E[E > 1.0] = 1.0

                return img_norm.reshape(c, h, w), H.reshape(c, h, w), E.reshape(c, h, w), C

            else:
                return img_norm.reshape(c, h, w)

    def convert_rgb2od(self, img):
        # Input: collapsed image of size (C, H*W) ranging from [0, 1]
        # Output: OD has size (C, H*W), while valid_idx has size (H*W, ).

        OD = -torch.log(img + torch.tensor(1 / self.Io))  # to make sure we do not get any -inf.
        valid_idx = torch.gt(torch.mean(OD, dim=0), self.beta)  # Index of valid, non-transparent pixels

        return OD, valid_idx

    def compute_matrices(self, img, HE_test=None, maxC_test=None):
        # This function calculates by default the following, in that order:
        #   (1) The stain vector of the current tile, HE_test.
        #   (2) The concentration of each stain, for each pixel of the tile, C.
        #   (3) The maximum concentration of each stain calculated over the whole tile, maxC_test.

        # Alternatively, the user can supply pre-calculated HE_test and maxC_test values if they do not want to infer
        # them on the current slide; this might be useful in the case where WSI-wise parameters are used.

        # Input: img: collapsed image of size (C, H*W) ranging from [0, 1]
        # Output: HE stain vector, concentrations of stains, maximum stain value.

        OD, ids = self.convert_rgb2od(img)  # OD has size C, H*W

        if OD[:, ids].shape[1] > 10:  # if slide can be processed (otherwise too many transparent pixels)

            eigvals, eigvecs = torch.linalg.eigh(cov(OD[:, ids]), UPLO='L')
            eigvecs = eigvecs[:, [1, 2]]  # eigenvalues are returned in ascending order, so take the last two.

            if HE_test is None:
                HE_test = self.find_HE(OD[:, ids], eigvecs)

            C = torch.linalg.lstsq(HE_test, OD)[0]  # determine concentrations of the individual stains

            if maxC_test is None:
                maxC_test = torch.stack([percentile(C[i, :], 99) for i in range(C.shape[0])])
        else:
            C = None

        return HE_test, C, maxC_test

    def find_HE(self, ODhat, eigvecs):
        # input: ODhat has shape (C, N), where N is the number of valid pixels for processing.
        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        That = torch.matmul(ODhat.T, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])
        minPhi = percentile(phi, self.alpha)
        maxPhi = percentile(phi, 100 - self.alpha)

        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin secondqw
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        return HE

    def fit(self, img):
        HE, _, maxC = self.compute_matrices(img)  # will fit HE, maxC on "img".
        self.HERef = HE
        self.maxCRef = maxC


########################################################################################################################

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
