# SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels
Code for IJCAI2022 [SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels](https://www.ijcai.org/proceedings/2022/455), SELC is a label correction method, it will automatically correct the noisy labels in training set. 

## Requirements
- Python 3.8.3
- Pytorch 1.8.1 


## Usage
For example, to train the model using SELC under class-conditional noise in the paper, run the following commands (Note that you need to download the CIFAR-10 and CIFAR-100 datasets first):
```train
python3 train_cifar_with_SELC.py
```
It can config with noise_mode, noise_rate, batch size and epochs. Similar commands can also be applied to other label noise scenarios.
### Hyperparameter options:
```
--data_path             path to the data directory
--noise_mode            label noise model(e.g. sym, asym)
--r                     noise level (0.0, 0.2, 0.4, 0.6, 0.8)
--loss                  loss functions (e.g. SELCLoss)
--alpha                 alpha in SELC
--batch_size            batch size
--lr                    learning rate
--lr_s                  learning rate schedule
--op                    optimizer (e.g. SGD)          
--num_epochs            number of epochs
```
For [ANIMAL-10N](https://dm.kaist.ac.kr/datasets/animal-10n/), [Clothing1M](https://github.com/Cysu/noisy_label) and [Webvision](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html) datasets, you need to download the datasets first and specify the data directory in the code. 

## Citing this work
If you use this code in your work, please cite the accompanying paper:
```
@article{lu2022selc,
  title={SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels},
  author={Lu, Yangdi and He, Wenbo},
  journal={IJCAI},
  year={2022}
}
```
