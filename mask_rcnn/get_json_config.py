from .demo import DemoConfig
#from coco import BagsConfig as DemoConfig 
import argparse

def get_demo_config(nclasses, mask_layer=True):
    
    config = DemoConfig(nclasses)
    config.BATCH_SIZE = 1
    config.IMAGES_PER_GPU = 1
    config.GPU_COUNT = 1
    config.MASK_LAYER = mask_layer
    
    return config

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('num_classes', help='Number of classes in the dataset', type=int)
    ap.add_argument('--use_mask_layer', help='Flag deciding if mask layer is used for training/inference', action='store_true')
    args = ap.parse_args()

    get_demo_config(args.num_classes, args.use_mask_layer).to_json()
