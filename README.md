# Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution (CoRPLE) 
This repo is the official implementation of,
**“Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution”**, 
Xingyuan Li, Jinyuan Liu*, Zhixin Chen, Yang Zou, Long Ma, Xin Fan, Risheng Liu, European Conference on Computer Vision __(ECCV)__, 2024.

[[pretrained models](https://drive.google.com/drive/folders/1lhk2MQX6JLE_t-QkJQ7aSZP_OvV4oP4k?usp=sharing)]

## 🤖 Download
Download our datasets of infrared image super-resolution with detection labels. Original images are provided by [TarDAL](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing).
- [Google Drive](https://drive.google.com/file/d/1h-v5vS6DnRBHI2XxpsFya-Li3rcWcXw1/view?usp=sharing)
  
Download our datasets of infrared image super-resolution with segmentation labels. Original images are provided by [SegMiF](https://drive.google.com/drive/folders/1T_jVi80tjgyHTQDpn-TjfySyW4CK1LlF?usp=sharing).
- [Google Drive](https://drive.google.com/file/d/1M8bKv8Z6CuOOR7g7hBYo36EToVkEy0Ly/view?usp=sharing)

## Dependencies

- Python 3.8
- PyTorch 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'CoRPLE'.
git clone https://github.com/hey-it-s-me/CoRPLE.git
conda create -n CoRPLE python=3.8
conda activate CoRPLE
pip install -r requirements.txt
python setup.py develop
```

## Training
- Run the following scripts. The training configuration is in `options/train/`.
  ```shell
  python basicsr/train.py -opt options/Train/train_CoRPLE_light_x4.yml
  ```
