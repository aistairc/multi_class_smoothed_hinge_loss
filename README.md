# Multi-class Smoothed Hinge Loss Function in Pre-training for Transfer Learning  
Project page of the paper 'Multi-class Smoothed Hinge Loss Function in Pre-training for Transfer Learning,' ICIP 2025. [paper](https://ieeexplore.ieee.org/document/11084603)


---
# How to use
### Dependencies
* pytorch >= 2.0

### Preparing
To download the pretrained weights, run
```
pip install huggingface_hub
python download.py
```
  
### Quick start

#### To use pre-trained resnet50 for transfer learning with cifar-100
```
python train_for_transfer.py -net resnet50 -pretrained resnet50_MCSH_m7.pth
```

#### To use pre-trained resnet50 for transfer learning with your own datasets
Please change line 142-156 to fit your own datsets.
```
python train_for_transfer.py -net resnet50 -pretrained resnet50_MCSH_m7.pth -num_classes {{your_dataset_class_num}} -dataset {{your_dataset_name}}
```

#### To use Multi-class Smoothed Hinge Loss Function in your code
change
```
loss = nn.CrossEntropyLoss()
```

in your code to
```
from MCSH_loss import MultiClassSmoothedHingeLoss
loss = MultiClassSmoothedHingeLoss(margin=YOUR_SETTING)
```
