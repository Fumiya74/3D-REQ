# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig
from .ecc import ECCDetectionDataset, ECCDatasetConfig
from .ecc1 import ECC1DetectionDataset, ECC1DatasetConfig 
#from .ecc04_per25 import ECC04P25DetectionDataset, ECC04P25DatasetConfig 
#from .ecc04_per50 import ECC04P50DetectionDataset, ECC04P50DatasetConfig
#from .ecc04_per0 import ECC04P0DetectionDataset, ECC04P0DatasetConfig
#from .ecc04_per100 import ECC04P100DetectionDataset, ECC04P100DatasetConfig
#from .ecc04_per90 import ECC04P90DetectionDataset, ECC04P90DatasetConfig
#from .ecc04_per99 import ECC04P99DetectionDataset, ECC04P99DatasetConfig

DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
    "ecc": [ECCDetectionDataset, ECCDatasetConfig],
    "ecc1": [ECC1DetectionDataset, ECC1DatasetConfig],
    #"ecc04": [ECC04DetectionDataset, ECC04DatasetConfig],
    #"ecc04_per0": [ECC04P0DetectionDataset, ECC04P0DatasetConfig],
    #"ecc04_per25": [ECC04P25DetectionDataset, ECC04P25DatasetConfig],
    #"ecc04_per50": [ECC04P50DetectionDataset, ECC04P50DatasetConfig],
    #"ecc04_per90": [ECC04P90DetectionDataset, ECC04P90DatasetConfig],
    #"ecc04_per99": [ECC04P99DetectionDataset, ECC04P99DatasetConfig],
    #"ecc04_per100": [ECC04P100DetectionDataset, ECC04P100DatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    dataset_dict = {
        "train": dataset_builder(dataset_config, split_set="train", root_dir=args.dataset_root_dir, augment=True),
        "test": dataset_builder(dataset_config, split_set="val", root_dir=args.dataset_root_dir, augment=False),
    }
    return dataset_dict, dataset_config
    
