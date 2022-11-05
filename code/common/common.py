from enum import Enum

class PathDatasets(Enum):
    # EXDARK = r"E:\dataset\ExDark"k
    # COCO_TEST = r"E:\dataset\MyCoco\test"
    # COCO_TRAIN = r"E:\dataset\MyCoco\train"
    # AUG_TEST = r"E:\dataset\augmentation_images"
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
    resnet50_finetuning = 1
    resnet50_finetuning_0_10 = 11
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

    resnet50_transfer = 11
    resnet50_transfer_1_9 = 2
    resnet50_transfer_2_8 = 3
    resnet50_transfer_3_7 = 4
    resnet50_transfer_4_6 = 5
    resnet50_transfer_5_5 = 6
    resnet50_transfer_6_4 = 7
    resnet50_transfer_7_3 = 8
    resnet50_transfer_8_2 = 9
    resnet50_transfer_9_1 = 10
