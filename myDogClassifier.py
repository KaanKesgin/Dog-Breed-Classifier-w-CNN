from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import random
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
Resnet50_model = ResNet50(weights='imagenet')
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_dataset(path):
    #function to load train, test and validation datasets
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 133)
    return files, targets

def path_to_tensor(img_path):
    #loads rgb image and returns a 4D tensor with shape (1,224,224,3)
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(Resnet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction>=151))

# load train, test, validation datasets for dogs
train_files, train_targets = load_dataset('../dogImages/train')
valid_files, valid_targets = load_dataset('../dogImages/valid')
test_files, test_targets = load_dataset('../dogImages/test')

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

#predict breed
from extract_bottleneck_features import *

bottleneck_featuresResnet50 = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_featuresResnet50['train']
valid_Resnet50 = bottleneck_featuresResnet50['valid']
test_Resnet50 = bottleneck_featuresResnet50['test']

modelResnet50 = Sequential()
modelResnet50.add(Flatten(input_shape=train_Resnet50.shape[1:]))
modelResnet50.add(Dense(1024, activation='relu'))
modelResnet50.add(Dropout(0.4))
modelResnet50.add(Dense(1024, activation='relu'))
modelResnet50.add(Dropout(0.4))
modelResnet50.add(Dense(133, activation='softmax'))

modelResnet50.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='dogResnet50.weights.best.hdf5', verbose=1,save_best_only=True)
modelResnet50.fit(train_Resnet50, train_targets, epochs=15, batch_size=50, validation_data=(valid_Resnet50, valid_targets), callbacks=[checkpointer], verbose=1, shuffle=True)

modelResnet50.load_weights('dogResnet50.weights.best.hdf5')

resnet50_predictions = [np.argmax(modelResnet50.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]
test_accuracy = 100*np.sum(np.array(resnet50_predictions)==np.argmax(test_targets, axis=1))/len(resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

def myModel_predict_breed(img_path):
    # extract bottleneck features\n",
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector\n",
    predicted_vector = modelResnet50.predict(bottleneck_feature)
    # return dog breed that is predicted by the model\n",
    return dog_names[np.argmax(predicted_vector)]

dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]

#testing the algorithm
print("That looks like a:")
print(myModel_predict_breed('testImages/segos.jpg'))
