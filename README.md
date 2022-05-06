# SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels
Code for IJCAI2022 [SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels](https://arxiv.org/pdf/2205.01156.pdf)

## Requirements
- Python 3.8.3
- Pytorch 1.8.1 


## Usage
For example, to train the model using SELC under class-conditional noise in the paper, run the following commands:
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


## Citing this work
If you use this code in your work, please cite the accompanying paper:
```
@article{lu2022selc,
  title={SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels},
  author={Lu, Yangdi and He, Wenbo},
  journal={arXiv preprint arXiv:2205.01156},
  year={2022}
}
```
