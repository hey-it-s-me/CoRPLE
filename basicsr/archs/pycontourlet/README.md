# pycontourlet
Contourlet transform toolbox

Modified from: https://github.com/mazayux/pycontourlet

Changes:
- Renamed `/pycontourlet` folder to `/pycontourlet4d` to support multi-channel PDFB decomposition on batches of images (original code only supports single channel, single imaage operation). The following changes are made:
  - Replaces `signal.convolve` with PyTorch's `F.conv2d`, which is faster and supports 4D tensors (batch, channel, height, width).
  - Rewrite `resampc.py` with Cython instead of weave, which is not supported in Python3.
- This result in around 15% speed up in training.