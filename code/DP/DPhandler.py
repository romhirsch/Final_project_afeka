import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
from enum import Enum
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from keras.callbacks import LearningRateScheduler
from sklearn import metrics
from enum import Enum
from common.common import *
import os
from collections.abc import Iterable
import augmentation.Augmenter as augmentation
from tensorflow.python.client import device_lib
from sklearn.utils import shuffle
import shutil
from augmentation.Dark_Augmenter import low_light_transform, blur, read_noise
from colorama import init
from termcolor import colored

init()
file_path = os.path.realpath(__file__)
file_path = file_path.replace(os.path.basename(file_path), '')

classes = ["bicycle", "boat", "bottle", "bus", "car",
           "cat", "chair", "cup", "dog", "motorcycle",
           "dining table", "person"]
classes = ['airliner',
 'bicycle',
 'Border collie',
 'cup',
 'dining table',
 'elephant',
 'folding chair',
 'golden retriever',
 'military plane',
 'minibus',
 'Persian cat',
 'school_bus',
 'speedboat',
 'tabby cat',
 'yawl']

print(tf.__version__)
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def uint8c(x):
    """custom uint8 - (if val < 0 then val = 0)"""
    x2 = np.copy(x)
    #x2[x2 < 0] = 0
    #x2[x2 > 255] = 255
    x2  = x2.clip(0, 255)
    return np.uint8(x2)

def preprocess(p=0.5):
    def _preprocess(x):
        if np.random.random() < p:
            alpha = np.random.uniform(0.8, 0.9)
            beta = np.random.uniform(0.5, 1)
            gamma = np.random.uniform(2, 9)
            var_noise = np.random.uniform(0.0001, 0.01)
            img_dark = low_light_transform(np.float32(x) / 255, alpha, beta, gamma)
            img_dark = blur(img_dark)
            x = read_noise(img_dark, var_noise)
            x = np.clip(x*255, 0, 255).astype(np.uint8)
           # aug = augmentation.Augmenter_rand(noise='gaussian', blur=(0, 300), saturation=False, gamma=(1.5, 5), brightness=False, contrast=False)
            #x = uint8c(x)
            #x = aug.illumination_augmenter(x)
        #     min_dim = min(x.shape[:2])
        #     _patch_size = sample(size_list, k=1)[0]
        #     patch_size = (_patch_size, _patch_size)
        #     x = extract_patches_2d(x, patch_size, max_patches=1, random_state=None)[0]
        #     x = np.array(Image.fromarray(x.astype('uint8'), 'RGB').resize((224, 224)))
        # x = x/255.0,.
            return preprocess_input(x)
        return preprocess_input(x)
    return _preprocess
#preprocess = preprocess(p=1)


class Environment(Enum):
    TRANSFER = 1
    FINETURNING = 2


class DPhandler(object):

    def __init__(self, classes, num_epochs, early_stop_patience=10, batch_training=100, batch_valid=100,
                 image_size=224, channels=3, finetuning=False, lr=0.001, name='dp',
                 checkpoint_path=r"E:\dataset\checkpoints", figure_path='', image_resize=224):
        self.name = name
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.classes = classes
        self.num_classes = len(classes)
        self._channels = channels
        self.image_size = image_size
        self.batch_training = batch_training
        self.batch_valid = batch_valid
        self.checkpoint_path = os.path.join(checkpoint_path, self.name)
        self.finetuning = finetuning
        self.lr = lr
        self.create_relavant_folders(figure_path)


    def create_relavant_folders(self, figure_path):
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.figure_path = figure_path
        if self.figure_path:
            if not os.path.exists(self.figure_path):
                os.mkdir(self.figure_path)


    def pretrain(self, model, weights, include_top=False):
        if include_top:
            self.pretrain_model = model(include_top=include_top, weights=weights)
            self.model = self.pretrain_model
        else:
             self.pretrain_model = model(include_top=include_top, pooling='avg', weights=weights)


    def create_model(self, classification_layers):
        self.model = Sequential()
        self.model.add(self.pretrain_model)
        for layer in classification_layers:
            self.model.add(layer)
        if self.finetuning:
            for layer in self.model.layers:
                layer.trainable = True
        else:
            self.model.layers[0].trainable = False
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        print('layers trainable:')
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name, layer.trainable)

    @staticmethod
    def Create_df_dataset_from_directories(directories, percent=1):
        if not isinstance(percent, Iterable):
            percent = [percent]*len(directories)
            df = pd.DataFrame({"Images": [], "Labels": []})
        for i, directory in enumerate(directories):
            images = []
            labels = []
            if os.path.exists(directory):
                sub_dirs = os.listdir(directory)
                sub_dirs = [v for v in sub_dirs if v in classes]
                for sub_dir in sub_dirs:
                    abs_path = os.path.join(directory, sub_dir)
                    image_list = os.listdir(abs_path)  # list of all image names in the directory
                    if percent[i] != 1:
                        image_number_choose  = int(len(image_list)*percent)
                        image_list = random.choices(image_list, k=image_number_choose)
                    image_list = list(map(lambda x: os.path.join(abs_path, x), image_list))
                    images.extend(image_list)
                    labels.extend([sub_dir] * len(image_list))
            df_temp = pd.DataFrame({"Images": images, "Labels": labels})
            df = pd.concat([df, df_temp])
        return df

    def loadDataset_df_preprocessing(self, df, preprocess):
        self.df_dataset = shuffle(df)
        data_generator = ImageDataGenerator(preprocessing_function=preprocess,
                                            validation_split=0.1,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        # flow_from_dataframe generates batches of augmented data
        # Both train & valid folders must have NUM_CLASSES sub-folders

        self.train_generator = data_generator.flow_from_dataframe(
            dataframe=self.df_dataset,
            directory=None,
            x_col="Images",
            y_col="Labels",
            subset="training",
            classes=self.classes,
            batch_size= self.batch_training,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(self.image_size, self.image_size))

        self.validation_generator = data_generator.flow_from_dataframe(
            dataframe=self.df_dataset,
            directory=None,
            x_col="Images",
            y_col="Labels",
            subset="validation",
            classes=self.classes,
            batch_size=self.batch_valid,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(self.image_size, self.image_size))

        self.step_per_epoch_train = len(self.train_generator)
        self.step_per_epoch_valid = len(self.validation_generator)
        print(self.batch_training, len(self.train_generator), self.batch_valid, len(self.validation_generator))



    def loadDataset_preprocessing(self, dataset_dir, preprocess):

        data_generator = ImageDataGenerator(preprocessing_function=preprocess,
                                            validation_split=0.2,
                                            horizontal_flip=True,
                                            vertical_flip=True)
        # flow_From_directory generates batches of augmented data
        # Both train & valid folders must have NUM_CLASSES sub-folders
        self.train_generator = data_generator.flow_from_directory(
            dataset_dir,
            classes=self.classes,
            target_size=(self.image_size, self.image_size),
            batch_size= self.batch_training,
            subset='training',
            shuffle=True,
            seed=42,
            class_mode='categorical')

        self.validation_generator = data_generator.flow_from_directory(
            dataset_dir,
            classes=self.classes,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_valid,
            subset='validation',
            seed=42,
            shuffle=True,
            class_mode='categorical')
        self.step_per_epoch_train = len(self.train_generator)
        self.step_per_epoch_valid = len(self.validation_generator)
        print(self.batch_training, len(self.train_generator), self.batch_valid, len(self.validation_generator))


    def fit(self, lr_sch=False):
        cb_early_stopper = EarlyStopping(monitor='val_loss', patience=self.early_stop_patience)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.checkpoint_path, f'{self.name}.best.h5'), #ModelCheckpoint
                                                         save_weights_only=True,
                                                         monitor='val_accuracy',
                                                         mode='max',
                                                         save_best_only=True)
        if lr_sch:
            lr_sched = LearningRateScheduler(lambda epoch: self.lr * (0.75 ** np.floor(epoch / 2)))
            call_backs = [cp_callback, cb_early_stopper, lr_sched]
        else:
            call_backs = [cp_callback, cb_early_stopper]

        self.fit_history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.step_per_epoch_train,
            epochs=self.num_epochs,
            validation_data=self.validation_generator,
            callbacks=call_backs,
            validation_steps=self.step_per_epoch_valid)
        # load best weights
        self.model.load_weights(os.path.join(self.checkpoint_path, f'{self.name}.best.h5'))


    def plot_acc_loss(self):
        plt.figure(1, figsize=(15, 8))
        plt.subplot(211)
        plt.plot(self.fit_history.history['accuracy'])
        plt.plot(self.fit_history.history['val_accuracy'])
        plt.title(f'{self.name} model accuracy and loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'])
        plt.subplot(212)
        plt.plot(self.fit_history.history['loss'])
        plt.plot(self.fit_history.history['val_loss'])
        #plt.title(f'{self.name} model loss')
        plt.ylabel(f'loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, f'{self.name}_acc_loss.png'))
        return np.max(self.fit_history.history['val_accuracy'])

    @staticmethod
    def load_dataset(dataset_dir, preprocess, classes, BATCH_SIZE=1, image_size=224):
        data_generator = ImageDataGenerator(preprocessing_function=preprocess)
        ds = data_generator.flow_from_directory(
            directory=dataset_dir,
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE,
            classes=classes,
            shuffle=False,
            seed=123
        )
        return ds


    def predict(self, ds):
        pred = self.model.predict(ds, steps=len(ds), verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        return pred, predicted_class_indices


    def confuction_mat(self, y_true, y_pred, class_names, ds_name):
        cm = confusion_matrix(y_true, y_pred)
        score = metrics.accuracy_score(y_true, y_pred)
        itemsCount = np.sum(cm, axis=1)
        predict = cm.diagonal()
        accuracy = predict / itemsCount
        class_names_with_acc = [class_name + ' ' + str(round(acc, 3)) for class_name, acc in zip(class_names, accuracy)]
        # Plot confusion matrix
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g')  # annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=10)
        ax.xaxis.tick_bottom()
        ax.set_ylabel('True', fontsize=20)
        ax.yaxis.set_ticklabels(class_names_with_acc, fontsize=10)
        plt.yticks(rotation=0)
        plt.title(f'{self.name} {ds_name} Confusion Matrix score={score}', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_path, f'{self.name}_{ds_name}.png'))
        plt.show(block=False)
        return accuracy, score

    def saved_model(self, path):
        self.model.save(path)


    def evaluate(self, ds, ds_name):
        pred, predicted_class_indices = self.predict(ds)
        #predicted_class_indices[predicted_class_indices==1].shape[0]/predicted_class_indices.shape[0]
        acc, score = self.confuction_mat(ds.labels, predicted_class_indices, list(ds.class_indices.keys()), ds_name)
        return np.round(acc, 3), np.round(score, 3)


    def plotLR(self):
        learning_rate = self.fit_history.history['lr']
        epochs = range(1, len(learning_rate) + 1)
        fig = plt.figure()
        plt.plot(epochs, learning_rate)
        plt.title('Learning rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        return (fig)


    @staticmethod
    def dshistc(ds):
        fig, ax = plt.subplots(1)
        ax.set_ylabel('ds')
        h1 = ax.hist(np.array(list(ds.class_indices.keys()))[ds.labels], bins=len(ds.class_indices.keys()))
        return h1


    def dshist(self):
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].set_ylabel('train ds')
        ax[1].set_ylabel('valid ds')
        h1 = ax[0].hist(np.array(list(self.train_generator.class_indices.keys()))[self.train_generator.labels])
        h2 = ax[1].hist(np.array(list(self.validation_generator.class_indices.keys()))[self.validation_generator.labels])
        return h1, h2


    def load_model(self, path):
        # Recreate the exact same model, including its weights and the optimizer
        self.model = tf.keras.models.load_model(path)
        # Show the model architecture
        self.model.summary()


def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im2)
    ax[1].hist(im2[...,0].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,1].flatten(), 256, [0, 256])
    ax[1].hist(im2[...,2].flatten(), 256, [0, 256])


def load_saved_model(path, ModelName, figure_path):
    if os.path.exists(path):
        dp = DPhandler(name=ModelName,
                  classes=classes,
                  num_epochs=20,
                  early_stop_patience=3,
                  batch_training=32,
                  batch_valid=32,
                  finetuning=True,
                  figure_path=figure_path)

        dp.load_model(path)
    else:
        print(f'saved model no exist {path}')
        return
    return dp


def train_model(ModelName, classes, weights,
                num_epochs, early_stop_patience,
                batch_training, batch_valid,
                pretrain_model, lr, lr_s, finetuning, figure_path, directories, preprocess, classificationLayers):
    dp = DPhandler(name=ModelName,
                   classes=classes,
                   num_epochs=num_epochs,
                   early_stop_patience=early_stop_patience,
                   batch_training=batch_training,
                   batch_valid=batch_valid,
                   finetuning=finetuning,
                   lr=lr,
                   figure_path=figure_path)
    dp.pretrain(pretrain_model, weights=weights)
    dp.create_model(classificationLayers)
    df_dataset = dp.Create_df_dataset_from_directories(directories)
    dp.loadDataset_df_preprocessing(df_dataset, preprocess)
    # df = pd.DataFrame({classes[i]:[df_dataset['Labels'][df_dataset['Labels']==classes[i]].count()] for i in range(len(classes))})
    #dp.loadDataset_preprocessing(dataset_dir, preprocess)
    dp.fit(lr_s)
    acc_val = dp.plot_acc_loss()
    return dp, acc_val

def update_df_summery(df_summery, dp, dataset_name, acc, acc_per_class):
    if np.any((df_summery['model'] == dp.name) & (df_summery['dataset'] == dataset_name)):
        #df_summery.loc[(df_summery['model'] == dp.name) & (df_summery['dataset'] == dataset_name), ['acc'] + classes] = [acc] + list(acc_per_class)
        df_summery.loc[len(df_summery), :] = [dp.name, dataset_name, 0, acc] + list(acc_per_class)

    else:
        try:
            df_summery.loc[len(df_summery), :] = [dp.name, dataset_name, 0, acc] + list(acc_per_class)
        except:
            pass
    return df_summery


def split_ds(path):
    df_ex = DPhandler.Create_df_dataset_from_directories(directories=[PathDatasets.EXDARK.value])
    path_train = r"E:\dataset\ExDark_train"
    path_test = r"E:\dataset\ExDark_test"
    p = 0.25
    for label in df_ex['Labels'].unique():
        df_temp = df_ex[df_ex['Labels'] == label]
        df_temp = shuffle(df_temp)
        n = df_temp.shape[0]
        ind = int(p*n)
        df_test = df_temp[:ind]
        df_train = df_temp[ind:]

        if not os.path.exists(os.path.join(path_train, label)):
            label_path_train = os.path.join(os.path.join(path_train, label))
            os.mkdir(label_path_train)
        if not os.path.exists(os.path.join(path_test, label)):
            label_path_test = os.path.join(os.path.join(path_test, label))
            os.mkdir(label_path_test)

        for path_img_test in df_test['Images']:
            file_test = os.path.basename(path_img_test)
            shutil.copyfile(path_img_test, os.path.join(label_path_test, file_test))

        for path_img_train in df_train['Images']:
            file_train = os.path.basename(path_img_train)
            shutil.copyfile(path_img_train, os.path.join(label_path_train, file_train))


def run_train(weights, probability_dark, ModelName,
              directories, num_epochs, batch_training,
              batch_valid, learning_rate, pretrain_model,
              lr_schedule, finetuning):
    mypreprocess = preprocess(p=probability_dark)
    figure_path = os.path.join(file_path, 'figures')
    figure_path = os.path.join(figure_path, ModelName)
    Path = r"E:\dataset\models_saved"
    saved_models_path = os.path.join(Path, ModelName + ".h")
    early_stop_patience = 3
    classificationLayers = [Flatten(), Dense(len(classes), activation='softmax')] #[Flatten(), Dense(512, activation='relu'), Dense(len(classes), activation='softmax')] #[layers.BatchNormalization(), layers.Dropout(0.2, name="top_dropout"), Dense(len(classes), activation='softmax')]#
    df_summery = pd.read_csv('dp_summery.csv', index_col='No')
    dp, acc_val = train_model(ModelName, classes, weights,
                              num_epochs, early_stop_patience,
                              batch_training, batch_valid, pretrain_model,
                              learning_rate, lr_schedule, finetuning, figure_path, directories, mypreprocess, classificationLayers)
    if np.any((df_summery['model'] == dp.name)):
        df_summery.loc[df_summery['model'] == dp.name, 'acc_val'] = np.round(acc_val, 2)
    else:
        try:
            df_summery.loc[len(df_summery), :] = [dp.name, '', acc_val, 0] + [0] * len(classes)
        except:
            pass
    dp.saved_model(saved_models_path)
    df_summery.to_csv('dp_summery.csv')
    return dp

if __name__ == '__main__':
    weights = 'imagenet'
    probability_dark = 0.5
    mypreprocess = preprocess(p=probability_dark)
    checkpoint_path = r"E:\dataset\checkpoints"
    directories = [PathDatasets.Imagenet_train.value]
    ModelName = Models.resnet50_transfer_5_5.name
    figure_path = os.path.join(file_path, 'figures')
    figure_path = os.path.join(figure_path, ModelName)
    Path = r"E:\dataset\models_saved"
    saved_models_path = os.path.join(Path, ModelName + ".h")
    num_epochs = 20
    early_stop_patience = 3
    batch_training = 32
    batch_valid = 32
    learning_rate = 0.0001
    pretrain_model = ResNet50 #EfficientNetB0
    do_train = 0
    lr_schedule = True
    finetuning = True

    # load test Datesets:
    #ds_train = DPhandler.load_dataset(dataset_train_dir, preprocess_input, classes)
    ds_test = DPhandler.load_dataset(PathDatasets.Imagenet_test.value, preprocess_input, classes)
    ds_aug1 = DPhandler.load_dataset(PathDatasets.Imagenet_aug1.value, preprocess_input, classes)
    ds_aug2 = DPhandler.load_dataset(PathDatasets.Imagenet_aug2.value, preprocess_input, classes)
    ds_aug3 = DPhandler.load_dataset(PathDatasets.Imagenet_aug3.value, preprocess_input, classes)
    ds_aug4 = DPhandler.load_dataset(PathDatasets.Imagenet_aug4.value, preprocess_input, classes)

    ds_aug = DPhandler.load_dataset(PathDatasets.Imagenet_aug.value, preprocess_input, classes)

    if finetuning:
        ModelNames = [Models.resnet50_finetuning_10_0.name, Models.resnet50_finetuning_9_1.name, Models.resnet50_finetuning_8_2.name, Models.resnet50_finetuning_7_3.name,
                     Models.resnet50_finetuning_6_4.name, Models.resnet50_finetuning_5_5.name, Models.resnet50_finetuning_4_6.name, Models.resnet50_finetuning_3_7.name,
                     Models.resnet50_finetuning_2_8.name, Models.resnet50_finetuning_1_9.name, Models.resnet50_finetuning_0_10.name]
    else:
        ModelNames = [Models.resnet50_transfer_10_0.name, Models.resnet50_transfer_9_1.name, Models.resnet50_transfer_8_2.name, Models.resnet50_transfer_7_3.name,
                     Models.resnet50_transfer_6_4.name, Models.resnet50_transfer_5_5.name, Models.resnet50_transfer_4_6.name, Models.resnet50_transfer_3_7.name,
                     Models.resnet50_transfer_2_8.name, Models.resnet50_transfer_9_1.name, Models.resnet50_transfer_0_10.name]

    # if do_train == 1 : #load our trained model
    #     dp = load_saved_model(saved_models_path, ModelName, figure_path)
    #     df_summery = pd.read_csv('dp_summery.csv', index_col='No')
    #     acc_per_class, acc = dp.evaluate(ds_aug1, "very_dark")
    #     df_summery = update_df_summery(df_summery, dp, "very_dark", acc, acc_per_class)
    #     df_summery.to_csv('dp_summery.csv')


    probability_darks = np.arange(0, 1.1, 0.1)
    for probability_dark, ModelName in zip(probability_darks, ModelNames):
        print(colored(f"{ModelName}",'green'))
        #probability_dark = 0.5
        if do_train == 1:  # load our trained model
            dp = load_saved_model(saved_models_path, ModelName, figure_path)
        else:
            dp = run_train(weights, probability_dark, ModelName,
                      directories, num_epochs, batch_training,
                      batch_valid, learning_rate, pretrain_model,
                      lr_schedule, finetuning)
        # evaluate accuracy and confusion matrix
        df_summery = pd.read_csv('dp_summery.csv', index_col='No')
        # acc_per_class, acc = dp.evaluate(ds_aug, "Dark_test")
        # df_summery = update_df_summery(df_summery, dp, "Dark_test", acc, acc_per_class)

        acc_per_class, acc = dp.evaluate(ds_aug1, "ds_aug1")
        df_summery = update_df_summery(df_summery, dp, "ds_aug1", acc, acc_per_class)

        acc_per_class, acc = dp.evaluate(ds_test, "test")
        df_summery = update_df_summery(df_summery, dp, "test", acc, acc_per_class)

        acc_per_class, acc = dp.evaluate(ds_aug2, "ds_aug2")
        df_summery = update_df_summery(df_summery, dp, "ds_aug2", acc, acc_per_class)

        acc_per_class, acc = dp.evaluate(ds_aug3, "ds_aug3")
        df_summery = update_df_summery(df_summery, dp, "ds_aug3", acc, acc_per_class)

        acc_per_class, acc = dp.evaluate(ds_aug4, "ds_aug4")
        df_summery = update_df_summery(df_summery, dp, "ds_aug4", acc, acc_per_class)

        df_summery.to_csv('dp_summery.csv')
        plt.close('all')

    # if do_train == 0: #train the model
    #     dp = run_train(weights, probability_dark, ModelName,
    #               directories, num_epochs, batch_training,
    #               batch_valid, learning_rate, pretrain_model,
    #               lr_schedule, finetuning)
    #
    # elif do_train == 1 : #load our trained model
    #     dp = load_saved_model(saved_models_path, ModelName, figure_path)
    #
    #
    #
    # # evaluate accuracy and confusion matrix
    # df_summery = pd.read_csv('dp_summery.csv', index_col='No')
    # acc_per_class, acc = dp.evaluate(ds_aug, "coco_aug_test")
    # df_summery = update_df_summery(df_summery, dp, "coco_aug_test", acc, acc_per_class)
    #
    # acc_per_class, acc = dp.evaluate(ds_test, "coco_test")
    # df_summery = update_df_summery(df_summery, dp, "coco_test", acc, acc_per_class)
    #
    # acc_per_class, acc = dp.evaluate(ds_dark, "exdark")
    # df_summery = update_df_summery(df_summery, dp, "exdark", acc, acc_per_class)
    #
    # df_summery.to_csv('dp_summery.csv')
    # plt.show()