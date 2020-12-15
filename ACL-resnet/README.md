# Adversarial Continual Learning 


This is the official PyTorch implementation of the [Adversarial Continual Learning](https://arxiv.org/abs/2003.09553) published at ECCV 2020. 


## Notice:
For the experiments shown in the main paper please refer to the [main directory](https://github.com/facebookresearch/Adversarial-Continual-Learning). This directory shall be used to use ACL with a ResNet18 backbone architecture. Follow the instructions [here](https://github.com/facebookresearch/Adversarial-Continual-Learning) to install the requirements and cloning the repo and use the following commands to run ACL with ResNet18 on CIFAR100 and miniImageNet from the current directory. If you have a different dataset, you can indeed create a Dataset class and corresponding dataloaders following how that is done for given datasets in `dataloaders` directory. 


Split CIFAR100 (20 Tasks):

``python main.py --config ./configs/config_cifar100.yml`` 

Split MiniImageNet (20 Tasks):

`python main.py --config ./configs/config_miniimagenet.yml`


## Authors:
[Sayna Ebrahimi](https://people.eecs.berkeley.edu/~sayna/) (UC Berkeley, FAIR), [Franziska Meier](https://am.is.tuebingen.mpg.de/person/fmeier) (FAIR), [Roberto Calandra](https://www.robertocalandra.com/about/) (FAIR), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (UC Berkeley), [Marcus Rohrbach](http://rohrbach.vision/) (FAIR)

### Citation
If using this code, parts of it, or developments from it, please cite our paper:
```
@article{ebrahimi2020adversarial,
  title={Adversarial Continual Learning},
  author={Ebrahimi, Sayna and Meier, Franziska and Calandra, Roberto and Darrell, Trevor and Rohrbach, Marcus},
  journal={arXiv preprint arXiv:2003.09553},
  year={2020}
}
```

#### Datasets

*miniImageNet* data should be [downloaded](https://github.com/yaoyao-liu/mini-imagenet-tools#about-mini-ImageNet) and pickled as a dictionary (`data.pkl`) with `images` and `labels` keys and placed in a sub-folder in `ags.data_dir` named as `miniimagenet`. The script used to split `data.pkl` into training and test sets is included in data dorectory (`data/`)

*notMNIST* dataset is included here in `./data/notMNIST` as it was used in our experiments. 

Other datasets will be automatically downloaded and extracted to `./data` if they do not exist.  

## Questions/ Bugs
* For questions/bugs, contact the author Sayna Ebrahimi via email sayna@berkeley.edu



## License
This source code is released under The MIT License found in the LICENSE file in the root directory of this source tree.


## Acknowledgements
Our code structure is inspired by [HAT](https://github.com/joansj/hat.).
