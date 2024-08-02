

## Parameters

All the parameters of the model are in configs/config.yml. You only need to change the base_path 

## Weights (for pretrained backbone ONLY)

The model can be trained with different backbones (resnet, xception, drn, mobilenet). The backbone weights are always pre-trained on ImageNet).


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

