import numpy as np
from skimage import io
from scipy.misc import imresize
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class ImageClassify:

    def __init__(self, class_names, image_size=100, learning_rate=0.001, test_split=0.1):
        self.model = None
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.classes = [class_name.lower() for class_name in class_names]
        self.image_data = []
        self.labels = []
        self.test_split = test_split

    def _extract_label(self, image_name):
        zeros = [0 for i in range(len(self.classes))]
        label_name = image_name.split('.')[0]
        index = self.classes.index(label_name.lower())
        zeros[index] = 1
        return zeros

    def _process_image(self, image):
        label = self._extract_label(image)
        img = io.imread(image)
        img = imresize(img, (self.image_size, self.image_size, 3))
        self.image_data.append(np.array(img))
        self.labels.append(np.array(label))

    def prepare_data(self, images):
        for image in images:
            self._process_image(image)

    def _image_to_array(self, image):
        img = io.imread(image)
        img = imresize(img, (self.image_size, self.image_size, 3))
        return img

    def build_model(self):
        convnet = input_data(shape=[None, self.image_size, self.image_size, 3], name='input')
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, len(self.classes), activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy',
                             name='targets')
        model = tflearn.DNN(convnet, tensorboard_dir='log')
        return model

    def train_model(self, model_name):
        X = self.image_data
        y = self.labels
        split = int(len(X) * self.test_split)
        X_train, X_test = X[split:], X[:split]
        y_train, y_test = y[split:], y[:split]
        model = self.build_model()
        model.fit(X_train, y_train, n_epoch=5, shuffle=True, validation_set=(X_test, y_test), show_metric=True,
                  batch_size=32)
        model.save(model_name)
        self.model = model

    def load_model(self, model_file):
        model = self.build_model()
        model.load(model_file)
        self.model = model

    def predict_image(self, image):
        img = self._image_to_array(image)
        results = self.model.predict([img])[0]
        most_probable = max(results)
        results = list(results)
        most_probable_index = results.index(most_probable)
        class_name = self.classes[most_probable_index]
        return class_name, results
