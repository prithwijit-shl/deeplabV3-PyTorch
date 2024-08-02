

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
