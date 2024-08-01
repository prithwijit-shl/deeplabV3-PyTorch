#from data_generators.datasets import cityscapes, coco, combine_dbs, pascal, sbd, deepfashion
from torch.utils.data import DataLoader
# from data_generators.deepfashion import DeepFashionSegmentation
from data_generators.topsalt import TopSalt

def initialize_data_loader(config):

    
    if config['dataset']['dataset_name'] == 'winchester':
        train_set = TopSalt(config, split='train')
        val_set = TopSalt(config, split='val')
        test_set = TopSalt(config, split='test')

    elif config['dataset']['dataset_name'] == 'all':
        train_set = TopSalt(config, split='train')
        val_set = TopSalt(config, split='val')
        test_set = TopSalt(config, split='test')

    else:
        raise Exception('dataset not implemented yet!')
    
    num_classes = train_set.num_classes
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['workers'], pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config['inference']['batch_size'], shuffle=False, num_workers=config['training']['workers'], pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes

