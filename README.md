

## Parameters

All the parameters of the model are in configs/config.yml.

## Weights (for pretrained backbone ONLY)

The trained weights can be found here:

https://drive.google.com/drive/folders/1O8KLZa1AABlLS6DlkkzHOgPqvT89GB_9?usp=sharing


The model can be trained with different backbones (resnet, xception, drn, mobilenet). The weights on the Drive has been trained with the ResNet backbone, so if you want to use another backbone you need to train from scratch (although the backbone weights are always pre-trained on ImageNet).


## Train

To train a model run:

```
python main.py -c configs/config.yml --train
```

You can set "weights_initialization" to "true" in config.yml, in order to restore the training after an interruption.  

During training the best and last snapshots can be stored if you set those options in "training" in config.yml.


## Inference 

To predict on the full test set run and get the metrics do: 

```
python main.py -c configs/config.yml --predict_on_test
```

