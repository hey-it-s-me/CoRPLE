import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_grayscale
from basicsr.archs.pycontourlet.pycontourlet4d.pycontourlet import batch_multi_channel_pdfbdec

def stack_same_dim(x):
    """Stack a list/dict of 4D tensors of same img dimension together."""
    # Collect tensor with same dimension into a dict of list
    output = {}
    
    # Input is list
    if isinstance(x, list):
        for i in range(len(x)):
            if isinstance(x[i], list):
                for j in range(len(x[i])):
                    shape = tuple(x[i][j].shape)
                    if shape in output.keys():
                        output[shape].append(x[i][j])
                    else:
                        output[shape] = [x[i][j]]
            else:
                shape = tuple(x[i].shape)
                if shape in output.keys():
                    output[shape].append(x[i])
                else:
                    output[shape] = [x[i]]
    else:
        for k in x.keys():
            shape = tuple(x[k].shape[2:4])
            if shape in output.keys():
                output[shape].append(x[k])
            else:
                output[shape] = [x[k]]
    
    # Concat the list of tensors into single tensor
    for k in output.keys():
        output[k] = torch.cat(output[k], dim=1)
        
    return output

class ContourletCNN(nn.Module):
    """C-CNN: Contourlet Convolutional Neural Networks.
    
    Based on: 
    
    Parameters
    ----------
    n_levs : list of int, default = [0, 3, 3, 3]
         The numbers of DFB (Directional Filter Bank) decomposition levels at
         each pyramidal level from coarse to fine-scale.
         
         In each pyramidal level, there is one number of DFB decomposition
         levels is denoted as `l`, resulting in `2^l = 8` wedge-shaped
         subbands in the frequency domain. When the DFB decomposition level is
         0, resulting in 1 subband in the frequency domain, which is effectively
         a 2-D wavelet decomposition.
         
         For example:
             n_levs = [0, 3, 3, 3]
             num_subbands = [1+3, 2^3, 2^3, 2^3]
         
    variant : {"origin", "SSFF", "SSF"}, \
            default="SSF"
        The variants of the Contourlet-CNN model. From left to right, each
        variant is an incremental version of the previous variant, as such
        in an abalation study in the original paper.
        ``"origin"``:
            The 'origin' splices the elongated decomposed images into its
            corresponding sizes since the contourlet has elongated supports.
            No SSF features is concatenated to the features in FC2 layer.
        ``"SSFF"``:
            Instead of splicing, the 'SSFF' (spatial–spectral feature fusion)
            via contourlet directly resize the elongated decomposed images
            into its corresponding sizes. No SSF features is concatenated to
            the features in FC2 layer.
        ``"SSF"``:
            In addition to 'SSFF', the 'SFF' (statistical feature fusion)
            that denotes the additional texture features of decomposed images,
            are concatenated to the features in FC2 layer.
            The mean and variance of each subbands are chosen as the texture
            features of decomposed images.
        
    spec_type : {"avg", "all"}, \
            default="all"
            The type of spectral information to obtain from the image.
            ``'avg'``:
                The spectral information is obtained from the average value of
                all image channel.
            ``'all'``:
                The spectral information is obtained from each channels in the
                image.
    References
    ----------
    DOI : 10.1109/TNNLS.2020.3007412
    
    """
    def __init__(self, num_classes, input_dim=(3, 224, 224), n_levs=[0, 3, 3, 3], variant="SSF", spec_type="all"):
        super(ContourletCNN, self).__init__()
        
        # Model hyperparameters
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.n_levs = n_levs
        self.variant = variant
        self.spec_type = spec_type
        
        # Conv layers parameters
        out_conv_1 = 64
        out_conv_2 = 64
        out_conv_3 = 128
        in_conv_4 = 128
        out_conv_4 = 128
        out_conv_5 = 256
        in_conv_6 = 256
        out_conv_6 = 256
        out_conv_7 = 256
        in_conv_8 = 256
        out_conv_8 = 256
        in_conv_9 = 256
        out_conv_9 = 128
        if spec_type == "avg":
            if variant == "origin":
                in_conv_2 = out_conv_1 + (2**n_levs[3]) // 2
                in_conv_3 = out_conv_2 + (2**n_levs[2]) // 2
                in_conv_5 = out_conv_4 + (2**n_levs[1]) // 2
                in_conv_7 = out_conv_6 + 4
            else:
                in_conv_2 = out_conv_1 + 2**n_levs[3]
                in_conv_3 = out_conv_2 + 2**n_levs[2]
                in_conv_5 = out_conv_4 + 2**n_levs[1]
                in_conv_7 = out_conv_6 + 4
        else:
            if variant == "origin":
                in_conv_2 = out_conv_1 + (2**n_levs[3] // 2) * input_dim[0]
                in_conv_3 = out_conv_2 + (2**n_levs[2] // 2) * input_dim[0]
                in_conv_5 = out_conv_4 + (2**n_levs[1] // 2) * input_dim[0]
                in_conv_7 = out_conv_6 + 4 * input_dim[0]
            else:
                in_conv_2 = out_conv_1 + 2**n_levs[3] * input_dim[0]
                in_conv_3 = out_conv_2 + 2**n_levs[2] * input_dim[0]
                in_conv_5 = out_conv_4 + 2**n_levs[1] * input_dim[0]
                in_conv_7 = out_conv_6 + 4 * input_dim[0]
                
        # Conv layers
        self.conv_1 = nn.Conv2d(input_dim[0], out_conv_1, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_conv_2, out_conv_2, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_conv_3, out_conv_3, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(in_conv_4, out_conv_4, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(in_conv_5, out_conv_5, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv2d(in_conv_6, out_conv_6, kernel_size=3, stride=2, padding=1)
        self.conv_7 = nn.Conv2d(in_conv_7, out_conv_7, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(in_conv_8, out_conv_8, kernel_size=3, stride=2, padding=1)
        self.conv_9 = nn.Conv2d(in_conv_9, out_conv_9, kernel_size=3, stride=1, padding=1)
        
        # Global average pooling (GAP) layer
        self.gap = nn.AvgPool2d(7)
        
        # Fully connected layers parameters
        in_fc1 = out_conv_9
        out_fc1 = 2048
        if variant == "SSF":
            if spec_type == "avg":
                in_fc2 = out_fc1 + 2 * (np.sum([2**i for i in n_levs[1:]]) + 4)
            else:
                in_fc2 = out_fc1 + 2 * (np.sum([2**i for i in n_levs[1:]]) + 4) * input_dim[0]
        else:
            in_fc2 = 2048
            
        # Fully connected layers
        self.fc_1 = nn.Linear(in_fc1, out_fc1)
        self.fc_2 = nn.Linear(in_fc2, num_classes)
        
    def __pdfbdec(self, x, method="resize"):
        """Pyramidal directional filter bank decomposition for a batch of
        images.

        Returns a list of 4D numpy array.

        Here's an example with an image with 3 channels, and batch_size=2:
            >>> self.n_levs = [0, 3, 3, 3]
            >>> coefs, sfs = self.__pdfbdec(x)
        This will yield:
            >>> coefs[0].shape
            (2, 24, 112, 112)
            >>> coefs[1].shape
            (2, 24, 56, 56)
            >>> coefs[2].shape
            (2, 24, 28, 28)
            >>> coefs[3].shape
            (2, 12, 14, 14)
            >>> sfs.shape
            (2, 168)
        """
        # Convert to from N-D channels to single channel by averaging
        if self.spec_type == 'avg':
            imgs = []
            # Iterate each image in a batch
            for i in range(x.shape[0]):
                # Convert to PIL and image and to grayscale image
                img = transforms.ToPILImage()(x[i])
                img = to_grayscale(img)
                imgs.append(img)
            # Restack and convert back to PyTorch tensor
            x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        coefs = batch_multi_channel_pdfbdec(x=x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 3, 3, 3], device=self.device)
        # coefs = batch_multi_channel_pdfbdec(x=x, pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 2, 2, 2],
        #                                     device=self.device)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Resize or splice
        if method == "resize":
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get maximum dimension (height or width)
                    max_dim = int(np.max((k[2], k[3])))
                    # Resize the channels
                    trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
                    coefs[k] = trans(coefs[k])
        else:
            for k in coefs.keys():
                # Resize if image is not square
                if k[2] != k[3]:
                    # Get minimum dimension (height or width)
                    min_dim = int(np.argmin((k[2], k[3]))) + 2
                    # Splice alternate channels (always even number of channels exist)
                    coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # Stack channels with same image dimension
        coefs = stack_same_dim(coefs)

        # Change coefs's key to number (n-1 to 0), instead of dimension
        for i, k in enumerate(coefs.copy()):
            idx = len(coefs.keys()) - i - 1
            coefs[idx] = coefs.pop(k)

        # Get statistical features (mean and std) for each image
        sfs = []
        for k in coefs.keys():
            sfs.append(coefs[k].mean(dim=[2, 3]))
            sfs.append(coefs[k].std(dim=[2, 3]))
        sfs = torch.cat(sfs, dim=1)

        return coefs, sfs
        
    def forward(self, x):
        
        # Perform PDFB decomposition to obtain the coefficients and it's statistical features
        if self.variant == "origin":
            coefs, _ = self.__pdfbdec(x, method="splice")
        else:
            coefs, sfs = self.__pdfbdec(x, method="resize")
        
        # AlexNet backbone convolution layers
        x = self.conv_1(x)
        x = self.conv_2(torch.cat((x, coefs[0].to(self.device)), 1))
        # x = self.conv_3(torch.cat((x, coefs[1].to(self.device)), 1))
        # x = self.conv_4(x)
        # x = self.conv_5(torch.cat((x, coefs[2].to(self.device)), 1))
        # x = self.conv_6(x)
        # x = self.conv_7(torch.cat((x, coefs[3].to(self.device)), 1))
        # x = self.conv_8(x)
        # x = self.conv_9(x)
        
        # Global average pooling layers
        x = self.gap(x)
        # Reshape to 1D
        x = x.view(x.size(0), x.size(1))
        
        # Fully connected layers
        x = self.fc_1(x)
        # Concat coefficient's statistical features if Statistical Feature Fusion
        # (SSF) is enabled
        if self.variant == "SSF":
            x = torch.cat((x, sfs.to(self.device)), 1)
        x = self.fc_2(x) 
        
        return x
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.device = args[0]
        return self
