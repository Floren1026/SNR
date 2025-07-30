# SNR
The code of "SNR：One Single Network for Image Steganography with Robust Post-Save Recovery"

# 🚀 Getting Started

## Dependencies and Installation
- Python 3.10.13 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 2.1.2](https://pytorch.org/) .

## Dataset
- In this paper, we use the commonly used dataset ImageNet, DIV2K, COCO and 102Flower.
- Run `sh scripts/train.py ` for training.
- Run `sh scripts/test.py ` for testing.

# Citation
If our work is useful for your research, please consider citing:

```
@article{yang2025snr,
  title={SNR: One Single Network for Image Steganography with Robust Post-Save Recovery},
  author={Yang, Chao and Wang, Shiyuan and Huang, Ying and Guo, Mingqiang},
  journal={Neurocomputing},
  pages={130929},
  year={2025},
  publisher={Elsevier}
}
```

# Acknowledgement
Part of our SNR framework is referred to [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [UDH](https://github.com/ChaoningZhang/Universal-Deep-Hiding). We thank all the contributors for open-sourcing.
