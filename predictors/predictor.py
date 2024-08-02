import os
import numpy as np
#import cv2
#from PIL import Image
import random
import datetime
import io
#from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import json
import time

from models.deeplab import *
from data_generators.deepfashion import DeepFashionSegmentation
from data_generators.topsalt import TopSalt
import argparse
from utils.datagen_utils import denormalize_image
#from deeplab_model.utils.plot_utils import centroid_histogram, mask_and_downsample, get_average_color, normalize_colors
from data_generators.data_generator import initialize_data_loader
from utils.metrics import Evaluator
from tqdm import tqdm
from losses.loss import SegmentationLosses
from losses.loss import DiceLoss
from utils.new_metrics import MeanIoU
import yaml
from PIL import Image
import matplotlib.colors as mcolors

class Predictor():
    def __init__(self, config,  checkpoint_path='experiments/checkpoint_best.pth.tar'):
        self.config = config
        self.checkpoint_path = checkpoint_path

#        with open(self.config_file_path) as f:

        self.categories_dict = {"background": 0, "top_salt": 1}

#        self.categories_dict = {"background": 0, "meningioma": 1, "glioma": 2, "pituitary": 3}
        self.categories_dict_rev = {v: k for k, v in self.categories_dict.items()}
        
        self.model = self.load_model()
        self.train_loader, self.val_loader, self.test_loader, self.nclass = initialize_data_loader(config)

        self.num_classes = self.config['network']['num_classes']
        self.evaluator = Evaluator(self.num_classes)
        self.iou = MeanIoU()
        # self.criterion = SegmentationLosses(weight=None, cuda=self.config['network']['use_cuda']).build_loss(mode=self.config['training']['loss_type'])
        self.dice_loss = DiceLoss(normalization='sigmoid')

    def load_model(self):
        model = DeepLab(num_classes=self.config['network']['num_classes'], backbone=self.config['network']['backbone'],
                        output_stride=self.config['image']['out_stride'], sync_bn=False, freeze_bn=True)


        if self.config['network']['use_cuda']:
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location={'cuda:0': 'cpu'})
            # checkpoint = torch.load(self.checkpoint_path, map_location={'cuda:0'})

#        print(checkpoint)
        model = torch.nn.DataParallel(model)

        model.load_state_dict(checkpoint['state_dict'])

        return model

    def inference_on_test_set(self):
        print("inference on test set")

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        count =0
        for i, sample in enumerate(tbar):
            count = count +1
            image, target = sample['image'], sample['label']
            if self.config['network']['use_cuda']:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                # print(output.shape())
            # loss = self.criterion(output, target)
            labels = target
            target = target.unsqueeze(1).float()
            loss = self.dice_loss(output, target)
            test_loss += loss.item()
            miou = self.iou(output, target)
            # print(image.shape)
            # print(output.shape)
            assert output.dim() == 4
            self.normalization = nn.Sigmoid()
            output = self.normalization(output)
            print_npy = output.cpu().numpy()
            result_npy = output
            # result = output
            output_npy = result_npy.long()
            prediction_npy = output_npy
            prediction_npy = prediction_npy.cpu().numpy()
            output_npy =output_npy.cpu().numpy()
            # result = output
            result = output > 0.01
            output = result.long()
            prediction = output
            prediction = prediction.cpu().numpy()
            image = image.cpu().numpy()
            labels = labels.cpu().numpy()
            mean_miou = []
            miou = miou.numpy()
            for j in range(image.shape[0]):
                fig, axs = plt.subplots(1, 3, figsize=(10, 10))
                images = []
                axs[0].set_title("Image")
                axs[1].set_title("Predictions " + "mIoU: "+ str(miou))
                axs[2].set_title("Ground Truth")
                image_new = denormalize_image(image[j].transpose(1, 2, 0))
                image_new *= 255.
                labels_new = labels[j]
                labels_new *= 255.
                images.append(axs[0].imshow(image_new.astype(int)))
                cmap = mcolors.ListedColormap(['black', 'red'])
                bounds = [0, 1, 2]  # Define the bounds for the colormap
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                images.append(axs[1].imshow(prediction[j].transpose(1, 2, 0), cmap=cmap, norm=norm))
                # images.append(axs[1].imshow(prediction[j].transpose(1, 2, 0), cmap=plt.get_cmap('nipy_spectral'), vmin=0, vmax=2))
                images.append(axs[2].imshow(labels_new.astype(int)))
                    # Construct the full path for saving the plot
                os.makedirs('predictions/images', exist_ok=True)
                os.makedirs('predictions/numpy', exist_ok=True)
                save_path = os.path.join('predictions/images', f'plot_{i}_{j}.png')
                save_path_npy = os.path.join('predictions/numpy', f'plot_{i}_{j}.npy')
                # print((print_npy[j].transpose(1, 2, 0)).shape)
                np.save(save_path_npy, (print_npy[j].transpose(1, 2, 0))[:,:,0])
                plt.savefig(save_path)  # Save each plot with a unique filename based on the loop index j
                plt.close(fig)  # Close the figure to release memory  
            mean_miou.append(miou)
        print("mean mIoU:", (sum(mean_miou)/len(mean_miou)))


    def segment_image(self, filename):

#        file_path = os.path.join(dir_path, filename)
        img = Image.open(filename).convert('RGB')

        sample = {'image': img, 'label': img}

        # sample = DeepFashionSegmentation.preprocess(sample, crop_size=513)
        sample = TopSalt.preprocess(sample, crop_size=513)

        # image, target = sample['image'], sample['label']
        # if self.config['network']['use_cuda']:
        #     image, target = image.cuda(), target.cuda()
        # with torch.no_grad():
        #     prediction = self.model(image)

        image, _ = sample['image'], sample['label']
        image = image.unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(image)

        image = image.squeeze(0).numpy()
        image = denormalize_image(np.transpose(image, (1, 2, 0)))
        image *= 255.

        prediction = prediction.squeeze(0).cpu().numpy()

#        print(prediction[])
        print(np.unique(prediction))
        prediction = np.argmax(prediction, axis=0)
        print(np.unique(prediction))
        return image, prediction
