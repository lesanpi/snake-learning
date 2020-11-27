import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np

class DQNetwork:

    def __init__(self, actions, input_shape, alpha = 0.1,
                 gamma = 0.99, dropout_prob = 0.1, load_path = '', logger = None):
        self.model = Sequential()
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.dropout_prob = dropout_prob

        self.model.add(BatchNormalization(axis = 1, input_shape = input_shape))
        self.model.add(Conv2D(32, 2, 2))
        self.model.add(ReLU())

        self.model.add(BatchNormalization(axis=1))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(ReLU())

        self.model.add(BatchNormalization(axis=1))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(ReLU())
        self.model.add(Flatten())

        self.model.add(Dropout(self.dropout_prob))
        self.model.add(Dense(512))
        self.model.add(ReLU())
        self.model.add(Dense(self.actions))

        self.optimizer = Adam()
        self.logger = logger

        if load_path != '':
            self.load(load_path)

        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer,
                           metrics=['accuracy'])

    def train(self, batch):
        x_train = []
        t_train = []

        for datapoint in batch:
            x_train.append(datapoint['source'].astype(np.float64))

            next_state_pred = self.predict(datapoint['dest'].astype(np.float64)).ravel()
            next_q_value = np.max(next_state_pred)

            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + (self.gamma * next_q_value)
            t_train.append(t)

        x_train = np.asarray(x_train).squeeze()
        t_train = np.asarray(t_train).squeeze()

        h = self.model.fit(x_train,
                           t_train,
                           batch_size=32,
                           nb_epoch=1)

        # Log loss and accuracy
        if self.logger is not None:
            self.logger.to_csv('loss_history.csv',
                               [h.history['loss'][0], h.history['acc'][0]])


    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size = 1)

    def save(self, filename = None):
        f = ('model.h5' if filename is None else filename)

        if self.logger is not None:
            self.logger.log(f'Saving model as {f}')
        self.model.save_weights(self.logger.path + f)

    def load(self, path):
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)