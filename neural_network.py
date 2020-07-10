import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tensorflow import keras

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.optimizer.set_jit(True)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


class NeuralNet:
    """
    Wrapper class for dynamically creating Deep Neural Networks (DNN)/Recurrent Neural Networks (RNN) using
    the functions provided by Tensorflow and Keras.
    """
    BUFFER_SIZE = 1000
    BATCH_SIZE = 256

    def __init__(self, data, features, timestep, target, training_ratio=0.8, predictions=3, observations=1, epochs=20, history=None):
        """
        Set the general parameters for the NeuralNet object
        :param data: Pandas DataTable with column names.
        :param features: The list of column names in data that will be used as features for training the data.
        :param timestep: The name of the column in data that contains the timestep variable.
        :param target: The name of the column in data that contains the target attribute, response.
        :param predictions: The number of predictions, timesteps, to predict.
        :param observations: The number of observations for each timestep
        """
        self.data = data
        self.features_considered = features
        self.timestep = timestep
        self.target = target
        self.predictions = predictions
        self.observations = observations
        self.training_ratio = training_ratio

        self.features = data[self.features_considered]
        self.features.index = data[self.timestep]
        self.dataset = self.features.values
        self.train_split = int(self.data.shape[0] * self.training_ratio)
        self.dataset_mean = self.dataset[:self.train_split].mean(axis=0)            # mean by column
        self.dataset_std = self.dataset[:self.train_split].std(axis=0)              # standard deviation by column
        self.dataset = (self.dataset - self.dataset_mean)/self.dataset_std          # standardize the dataset values

        self.future_target = self.predictions * self.observations
        self.past_history = int(len(self.dataset) - self.train_split - self.future_target - 1) \
            if history is None \
            else history

        self.x_train_multi, self.y_train_multi = None, None
        self.x_val_multi, self.y_val_multi = None, None
        self.train_data_multi, self.val_data_multi = None, None

        self.initialize()

        self.epochs = epochs
        self.evaluation_interval = self.train_split

        self.model = None
        self.multi_step_history = None

    def multivariate_data(self, start_index, end_index, history_size, target_size, step):
        data_target = self.dataset[:, 1]
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(self.dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(self.dataset[indices])
            labels.append(data_target[i:i + target_size])

        return np.array(data), np.array(labels)

    def create_time_steps(self, length):
        return list(range(-length, 0))

    def plot_train_history(self, history, title):
        """
        Training loss << validation loss: Overfitting
        Training loss >> validation loss: Underfitting
        Training loss ~ validation loss: Goal
        :param history:
        :param title:
        :return:
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))

        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()
        plt.show()

    def multi_step_plot(self, history, true_future, prediction):
        plt.figure(figsize=(12, 6))
        num_in = self.create_time_steps(len(history))
        num_out = len(true_future)

        plt.plot(num_in, np.array(history[:, 1]), label='History')
        plt.plot(np.arange(num_out) / self.future_target, np.array(true_future), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out) / self.future_target, np.array(prediction), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.show()

    def initialize(self):
        self.x_train_multi, self.y_train_multi = self.multivariate_data(0, self.train_split, self.past_history,
                                                              self.future_target, self.observations)
        self.x_val_multi, self.y_val_multi = self.multivariate_data(self.train_split, None, self.past_history,
                                                          self.future_target, self.observations)

        self.train_data_multi = tf.data.Dataset.from_tensor_slices((self.x_train_multi, self.y_train_multi))
        self.train_data_multi = self.train_data_multi.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).repeat()

        self.val_data_multi = tf.data.Dataset.from_tensor_slices((self.x_val_multi, self.y_val_multi))
        self.val_data_multi = self.val_data_multi.batch(self.BATCH_SIZE).repeat()

    def define_model(self, layers):
        self.model = tf.keras.models.Sequential()
        layer_count = len(layers)
        current_layer = 1
        for l in layers:
            layer = getattr(tf.keras.layers, l["name"])
            if layer_count != current_layer or layer_count == 1:
                self.model.add(layer(
                    int(l["nodes"]),
                    input_shape=self.x_train_multi.shape[-2:],
                    **l["kwargs"])
                )
            else:
                self.model.add(layer(
                    int(l["nodes"]),
                    **l["kwargs"])
                )
            current_layer += 1
        self.model.add(tf.keras.layers.Dense(self.future_target))

    def define_optimizer(self, name, loss_function, kwargs):
        optimizer = getattr(tf.keras.optimizers, name)
        self.model.compile(optimizer=optimizer(**kwargs), loss=loss_function, metrics=['accuracy'])

    def get_model_details(self):
        return self.model.summary()

    def train_model(self, early_stop=True, early_stop_monitor='val_loss', early_stop_patience=10):
        early_stop_function = [keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=early_stop_patience)]
        if not early_stop:
            early_stop_function = None

        self.multi_step_history = self.model.fit(
            self.train_data_multi,
            epochs=self.epochs,
            steps_per_epoch=self.evaluation_interval,
            validation_data=self.val_data_multi,
            validation_steps=50,
            callbacks=early_stop_function
        )
        self.plot_train_history(self.multi_step_history, 'Multi-Step Training and validation loss')

        for x, y in self.val_data_multi.take(1):
            self.multi_step_plot(x[0], y[0], self.model.predict(x)[0])

    def get_predictions(self):
        predictions = []
        for x, y in self.val_data_multi.take(1):
            predictions.append(self.model.predict(x)[0])
        return predictions


if __name__ == "__main__":
    # ------------ Data input and setup configuration ---------- #
    _raw_data = pd.read_excel(os.path.join("data", "VB_Data_1a.xlsx"))                  # Data source
    _features_considered = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']       # Columns in data source to use as feature attributes
    _timestamp = 'ID'                                                                   # Column in data source to use as timestep value
    _target = 'Response'                                                                # Column in data source to use as target attribute

    # ----------- Define Hyper-parameters -------------- #
    # Layers define the neural network layers
    # name: A valid tensorflow keras layer. https://www.tensorflow.org/api_docs/python/tf/keras/layers
    # nodes: The number of nodes for the layer
    # activation_function: The activation function to use for the layer. https://www.tensorflow.org/api_docs/python/tf/keras/activations
    _layers = [
        {"name": "LSTM", "nodes": 8, "kwargs": {"activation": "swish", "return_sequences": True}},
        {"name": "Dropout", "nodes": 0.2, "kwargs": {}},
        {"name": "GRU", "nodes": 32, "kwargs": {"activation": "tanh", "return_sequences": True}},
        {"name": "LSTM", "nodes": 8, "kwargs": {"activation": "swish"}}
    ]
    # _layers = [
    #     {"name": "GRU", "nodes": 32, "kwargs": {"activation": "tanh", "return_sequences": True}},
    #     {"name": "GRU", "nodes": 32, "kwargs": {"activation": "tanh"}}
    # ]
    # Optimizer to be used by the neural network.
    # name: A valid tensorflow keras optimizer. https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    # kwargs: The arguments for the specific
    # rho: Discounting factor for gradient, defaults to 0.9
    # loss_function: The loss function for the optimizer. https://www.tensorflow.org/api_docs/python/tf/keras/losses
    _optimizer = {"name": "RMSprop", "loss_function": "mae", "kwargs": {"learning_rate": 0.002, "rho": 0.85}}
    # _optimizer = {"name": "Adam", "loss_function": "mse", "kwargs": {"learning_rate": 0.001}}

    # Create Neural Net class object, initializing all configuration settings.
    # Additional arguments (with default values): training_ratio=0.8, predictions=3, observations=1, epochs=20
    rnn = NeuralNet(_raw_data, _features_considered, _timestamp, _target)
    rnn.define_model(_layers)
    rnn.define_optimizer(_optimizer["name"], _optimizer["loss_function"], _optimizer["kwargs"])
    rnn.get_model_details()     # Returns the configuration and layered structure of the neural network
    rnn.train_model(early_stop=False)

    results = rnn.get_predictions()
    print(results)
