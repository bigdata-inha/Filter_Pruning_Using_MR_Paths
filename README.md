# Structured pruning using max relevance paths from a pretrained neural networks


comarison mehtods
  
Thinet(option : greedy) paper : ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression(ICCV,2017)

LASSO(option : lasso) paper: Channel Pruning for Accelerating Very Deep Neural Networks(ICCV,2017)


# Prerequistes

pytorch, python : pytorch 1.6 ↑, python 3.7 ↑
package : numpy, os, torchsummaryX, tqdm, networkx


## have to download VGG16_BN weight and then move model weights to directory './experiments/vgg16_exp_cifar100_0/checkpoints/' 

https://drive.google.com/drive/folders/1D0zxgCMg3nDUGGxCt8n2PDtcU6FHVysP?usp=sharing


# Experiments
VGG16-BN / CIFAR-100 



<img src="https://user-images.githubusercontent.com/46774714/101281372-dee3c280-3811-11eb-9a0a-f8164a120b7b.jpg" width="100%" height="90%">

<img src="https://user-images.githubusercontent.com/46774714/101281328-a04e0800-3811-11eb-84f9-1d6df1a59085.jpg" width="100%" height="90%">
