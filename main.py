import argparse
import os
import numpy as np

import torch
import yaml

from trainers.trainer import Trainer
from predictors.predictor import Predictor
from PIL import Image
from matplotlib import pyplot as plt

def train(args):

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()
    config['checkname'] = 'deeplab-'+str(config['network']['backbone'])

#    torch.manual_seed(config['seed'])
    trainer = Trainer(config)
        
    print('Starting Epoch:', trainer.config['training']['start_epoch'])
    print('Total Epoches:', trainer.config['training']['epochs'])
    
    for epoch in range(trainer.config['training']['start_epoch'], trainer.config['training']['epochs']):
        trainer.training(epoch)
        if not trainer.config['training']['no_val'] and epoch % config['training']['val_interval'] == (config['training']['val_interval'] - 1):
            trainer.validation(epoch)

    trainer.writer.close()

def predict_on_test_set(args):
#    print("predict on test")

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()

    predictor = Predictor(config, checkpoint_path='/glb/hou/pt.sgs/data/ml_ai_us/uspcjc/models/deeplabV3SIMCLR/experiments/checkpoint_best.pth.tar')

    predictor.inference_on_test_set()

def predict(args):
#    print("predict")

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    filename = args.filename

#    print(filename)

    config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()

    predictor = Predictor(config, checkpoint_path='/glb/hou/pt.sgs/data/ml_ai_us/uspcjc/models/deeplabV3SIMCLR/experiments/checkpoint_best.pth.tar')

    image, prediction = predictor.segment_image(filename)
    return image, prediction
#    print(np.max(prediction))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Seq2seq')
    parser.add_argument('-c', '--conf', help='path to configuration file', required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')    
    group.add_argument('--predict_on_test_set', action='store_true', help='Predict on test set')
    group.add_argument('--predict', action='store_true', help='Predict on single file')

    parser.add_argument('--filename', help='path to file')
    
    args = parser.parse_args()


    if args.predict_on_test_set:
        predict_on_test_set(args)      

    elif args.predict:
        if args.filename is None:
            raise Exception('missing --filename FILENAME')
        else:
            image, prediction = predict(args)
            fig, axs = plt.subplots(1, 2, figsize=(16, 16))

            images = []

            axs[0].set_title("Image")
            axs[1].set_title("Predictions")

            images.append(axs[0].imshow(image.astype(int)))
            # images.append(axs[1].imshow(annotation, cmap=plt.get_cmap('nipy_spectral'), vmin=0, vmax=num_classes))
            images.append(axs[1].imshow(prediction, cmap=plt.get_cmap('nipy_spectral'), vmin=0, vmax=2))
            cbar = fig.colorbar(images[1], ax=axs, orientation='horizontal', ticks=[x for x in range(2)], fraction=.1)
            # cbar.ax.set_xticklabels(list(predictor.categories_dict.keys()), rotation=55)

            plt.show()

    elif args.train:
        print('Starting training')
        train(args)   
    else:
        raise Exception('Unknown args') 