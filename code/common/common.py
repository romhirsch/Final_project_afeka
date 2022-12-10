from enum import Enum


class PathDatasets(Enum):
    # EXDARK = r"E:\dataset\ExDark"k
    # COCO_TEST = r"E:\dataset\MyCoco\test"
    # COCO_TRAIN = r"E:\dataset\MyCoco\train"
    # AUG_TEST = r"E:\dataset\augmentation_images"
    Imagenet_aug1 = r"E:\Imagenet_aug_1"
    Imagenet_aug2 = r"E:\Imagenet_aug_2"
    Imagenet_aug3 = r"E:\Imagenet_aug_3"
    Imagenet_aug4 = r"E:\Imagenet_aug_4"
    lol_low = r"E:\dataset\lol\low"
    lol_high = r"E:\dataset\lol\high"
    real = r"E:\real_images"
    Imagenet_aug_very_low = r"E:\Imagenet_aug_very_low"
    Imagenet_aug = r"E:\Imagenet_aug"
    Imagenet_test = r"E:\imagenet_test"
    Imagenet_train = r"E:\imagenet_train"
    Imagenet_orig = r"E:\imagenet_orig"
    COCO_TEST_SMALL = r"E:\dataset\MyCoco\test_small"
    COCO_TRAIN_SMALL = r"E:\dataset\MyCoco\train_small"
    EXDARK_TRAIN = r"E:\dataset\ExDark_train"
    EXDARK_TEST = r"E:\dataset\ExDark_test"
    COCO_AUG_TEST = r"E:\dataset\augmentation_images_small"
    EXDARK = r"E:\dataset\ExDark"
    COCO_TEST = r"E:\dataset\MyCoco\test"
    COCO_TRAIN = r"E:\dataset\MyCoco\train"
    AUG_TEST = r"E:\dataset\augmentation_images"

class Models(Enum):
    #dark: coco
    resnet50_finetuning = 0
    resnet50_finetuning_0_10 = 1
    resnet50_finetuning_1_9 = 2
    resnet50_finetuning_2_8 = 3
    resnet50_finetuning_3_7 = 4
    resnet50_finetuning_4_6 = 5
    resnet50_finetuning_5_5 = 6
    resnet50_finetuning_6_4 = 7
    resnet50_finetuning_7_3 = 8
    resnet50_finetuning_8_2 = 9
    resnet50_finetuning_9_1 = 10
    resnet50_finetuning_10_0 = 11


    resnet50_transfer_0_10 = 12
    resnet50_transfer_1_9 = 13
    resnet50_transfer_2_8 = 14
    resnet50_transfer_3_7 = 15
    resnet50_transfer_4_6 = 16
    resnet50_transfer_5_5 = 17
    resnet50_transfer_6_4 = 18
    resnet50_transfer_7_3 = 19
    resnet50_transfer_8_2 = 20
    resnet50_transfer_9_1 = 21
    resnet50_transfer_10_0 = 22
