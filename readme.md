<div align="center">

# 🤖🧠 Training ViT with Limited Data for Alzheimer's Disease Classification: an Empirical Study

  Code implementation of the empirical study paper
</div>

## Introduction
In this paper, we thoroughly investigate the application of a Vision Transformer (ViT) in brain medical imaging in a low-data regime. We conducted an intensive set of experiments using a limited amount of labeled brain 3D MRI data for the task of Alzheimer's disease classification. As a result, our experiments yield an optimal training recipe, thus paving the way for Vision Transformer-based models for other low-data medical imaging applications. To bolster further development, we release our assortment of pre-trained models for other MRI applications

## Required packages:
```
cuda: 11.7.1
numpy: 1.23.5
monai: 1.1.0
python: 3.10.9
pytorch: 1.13
scikit-image: 0.19.3
```
## Pre-trained models
We share an assortment of our pre-trained models for interested researchers as starting weights for training the ViT-B models using MRI images for other applications.
We believe that this open-source research will bolster the development of medical AI applications.

### Our pre-trained models:
| Pre-trained model    | Pre-training datasets   | ADNI1 score | ADNI2 score | Link to download |
| ---------------------|------------------------ |-------------|-------------|------------------|
| *ViT-B 75% mask ratio | BRATS 2023, IXI, OASIS3 |  79.6       | 81.9        | [Download](https://drive.google.com/file/d/1vSxBZ78NXdcAklyFtJPOHtP2ttpUMTQ8/view?usp=sharing) |
| ViT-B 50% mask ratio | BRATS 2023, IXI, OASIS3 |  79.5       | 79.2        | [Download](https://drive.google.com/file/d/1RZYlnCh1Gac5_t6UleBJbeVw5Wh3S75b/view?usp=drive_link) |
| ViT-B 25% mask ratio | BRATS 2023, IXI, OASIS3 |  74.5       | 74.0        | [Download](https://drive.google.com/file/d/1qF2kykBFpEBpU7cAIAh8_9PjXe6V2KtP/view?usp=drive_link) |
| ViT-B 75% mask ratio | BRATS 2023, IXI, OASIS3, HCP | 79.5   | 80.8        | [Download](https://drive.google.com/file/d/1Tsefm8gtpp1XDMAQnufqJHFWV2CtF3Tw/view?usp=drive_link) |
| ViT-B 75% mask ratio | BRATS 2023, IXI         | 77.9        | 81.9        | [Download](https://drive.google.com/file/d/1Q9_JZEWjTM7-7c0MURo0SAxVAC-e2b6g/view?usp=drive_link) |
| ViT-B 75% mask ratio | BRATS 2023              | 75.9        |  76.8       | [Download](https://drive.google.com/file/d/1T-Rx0T6dKnMYXMRscb0yV62j-wjzc1ga/view?usp=drive_link) |
| **ViT-B with distillation token 75% mask ratio | BRATS 2023, IXI, OASIS3 | 79.7   | 82.0  | [Download](https://drive.google.com/file/d/1pUqey6QOKJThmEzzeMrC8rCujuU9qdF2/view?usp=drive_link) |
| **ViT-B with distillation token 75% mask ratio | BRATS 2023, IXI, OASIS3, HCP | 79.9   | 80.2  | [Download](https://drive.google.com/file/d/13323d-g_FzRQtTlVfROskiYVDm3KsxGY/view?usp=drive_link) |

> (*) Our recommendation: the best model in this comparison in terms of performance and computational cost.

> (**) The _distillation token-based ViT-B_ was trained using a teacher model _3D ResNet-152_, obtained from [MONAI](https://github.com/Project-MONAI/MONAI). This setup achieved an average accuracy of **84.63% for ADNI2** and **82.51% for ADNI1**.

### How to use our pre-trained models
Below is the sample code for loading pre-trained weights into a PyTorch model. This code handles the loading of the model checkpoint and loads the weights into your model architecture.

```python
import torch

# Load the pre-trained model checkpoint
pre_trained_model_path = 'path_to_your_pretrained_model.pth'
checkpoint = torch.load(pre_trained_model_path, map_location='cpu')
print("Loaded pre-trained checkpoint from: %s" % pre_trained_model_path)

# Extract the state dictionary from checkpoint
checkpoint_model = checkpoint['net']

# Load the state dict into your model
model = YourModel()  # replace YourModel with your actual model class
msg = model.load_state_dict(checkpoint_model, strict=False)

# Handling possible mismatches
if msg.missing_keys:
    print("Warning: Missing keys in state dict: ", msg.missing_keys)
if msg.unexpected_keys:
    print("Warning: Unexpected keys in state dict: ", msg.unexpected_keys)
```

## How to cite:

If you use this repository in your work, please cite the following paper:

```bibtex
@InProceedings{ViT_recipe_for_AD,
  author = {Kunanbayev, Kassymzhomart and Shen, Vyacheslav and Kim, Dae-Shik},
  title = {Training ViT with Limited Data for Alzheimer’s Disease Classification: an Empirical Study},
  booktitle = {Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
  year = {2024},
  publisher = {Springer Nature Switzerland},
  volume = {LNCS 15012},
  month = {October},
  page = {pending}
}
