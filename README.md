# Contourlet Refinement Gate Framework 
This repo is the implementation of,
**“Contourlet Refinement Gate Framework for Thermal Spectrum Distribution Regularized Infrared Image Super-Resolution”**, 
Yang Zou, Zhixin Chen, Zhipeng Zhang, Xingyuan Li, Long Ma, Jinyuan Liu, Peng Wang, Yanning Zhang.

[[pretrained models](https://drive.google.com/drive/folders/1lhk2MQX6JLE_t-QkJQ7aSZP_OvV4oP4k?usp=sharing)]
[[arXiv link](https://arxiv.org/pdf/2411.12530)]

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
git clone https://github.com/hey-it-s-me/CRG.git
conda create -n CRG python=3.8
conda activate CRG
pip install -r requirements.txt
python setup.py develop
```

## Training
- Run the following scripts. The training configuration is in `options/train/`.
  ```shell
  python basicsr/train.py -opt options/Train/train_CRG_light_x2.yml
  python basicsr/train.py -opt options/Train/train_CRG_light_x4.yml
  ```
- The training experiment is in `experiments/`.
  
## Testing
- Run the following scripts. The testing configuration is in `options/test/`.
  ```shell
  python basicsr/train.py -opt options/Test/my_test_CRG_light_x2.yml
  python basicsr/train.py -opt options/Test/my_test_CRG_light_x4.yml
  ```
- The output is in `results/`.

## Acknowledgements

This code is built on  [DAT](https://github.com/zhengchen1999/DAT.git), [Contourlet-CNN
](https://github.com/xKHUNx/Contourlet-CNN), and [FasterViT](https://github.com/NVlabs/FasterViT).

## Citation

If this work has been helpful to you, please feel free to cite our paper!

```
@article{zou2024contourlet,
  title={Contourlet Refinement Gate Framework for Thermal Spectrum Distribution Regularized Infrared Image Super-Resolution},
  author={Zou, Yang and Chen, Zhixin and Zhang, Zhipeng and Li, Xingyuan and Ma, Long and Liu, Jinyuan and Wang, Peng and Zhang, Yanning},
  journal={arXiv preprint arXiv:2411.12530},
  year={2024}
}
```
