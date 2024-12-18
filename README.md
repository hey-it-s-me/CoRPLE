# Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution (CoRPLE) 
This repo is the official implementation of,
**“Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution”**, 
Xingyuan Li, Jinyuan Liu*, Zhixin Chen, Yang Zou, Long Ma, Xin Fan, Risheng Liu, European Conference on Computer Vision __(ECCV)__, 2024.

[[pretrained models](https://drive.google.com/drive/folders/1lhk2MQX6JLE_t-QkJQ7aSZP_OvV4oP4k?usp=sharing)]
[[paper link](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00391.pdf)]


## Updates

### New Version: `CRG` Branch

The implementation of the 'Contourlet Refinement Gate Framework' is available on the [`CRG` branch](https://github.com/hey-it-s-me/CoRPLE/tree/CRG). 

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
  python basicsr/train.py -opt options/Train/train_CoRPLE_light_x2.yml
  python basicsr/train.py -opt options/Train/train_CoRPLE_light_x4.yml
  ```
- The training experiment is in `experiments/`.
  
## Testing
- Run the following scripts. The testing configuration is in `options/test/`.
  ```shell
  python basicsr/train.py -opt options/Test/my_test_CoRPLE_light_x2.yml
  python basicsr/train.py -opt options/Test/my_test_CoRPLE_light_x4.yml
  ```
- The output is in `results/`.

## Acknowledgements

This code is built on  [DAT](https://github.com/zhengchen1999/DAT.git) and [Contourlet-CNN
](https://github.com/xKHUNx/Contourlet-CNN).

## Citation

If this work has been helpful to you, please feel free to cite our paper!

```
@inproceedings{li2024contourlet,
  title={Contourlet residual for prompt learning enhanced infrared image super-resolution},
  author={Li, Xingyuan and Liu, Jinyuan and Chen, Zhixin and Zou, Yang and Ma, Long and Fan, Xin and Liu, Risheng},
  booktitle={European Conference on Computer Vision},
  pages={270--288},
  year={2024},
  organization={Springer}
}
```
