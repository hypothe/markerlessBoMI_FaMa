# Python ≥3.5 is required
import queue
import sys
assert sys.version_info >= (3, 5)
#
# # Scikit-Learn ≥0.20 is required
# import sklearn
# assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0-preview is required
import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Add, Multiply
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from keras.losses import  mse

# assert tf.__version__ >= "2.0"

# Common imports
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
from scripts.reaching import Reaching
import scripts.reaching_functions as reaching_functions
import matplotlib.pyplot as plt

## utility functions 
def sigmoid(x, L=1, k=1, x0=0, offset=0):
  return offset + L / (1 + math.exp(k*(x0-x)))

def doubleSigmoid(x):
    if x < 0:
        return sigmoid(x, L=0.5, k=12, x0=-0.5, offset=-0.5)
    else:
        return sigmoid(x, L=0.5, k=12, x0=0.5, offset=0.)

def compute_vaf(x, x_rec):
    """
        computes the VAF between original and reconstructed signal
        :param x: original signal
        :param x_rec: reconstructed signal
        :return: VAF
    """
    x_zm = x - np.mean(x, 0)
    x_rec_zm = x_rec - np.mean(x_rec, 0)
    vaf = 1 - (np.sum(np.sum((x_zm - x_rec_zm) ** 2)) / (np.sum(np.sum(x_zm ** 2))))
    return vaf * 100


def temporalize(X, lookback):
    '''
    A UDF to convert input data into 3-D
    array as required for LSTM (and CNN) network.
    '''

    output_X = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1, lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
    return output_X


class LossCallback(keras.callbacks.Callback):
    """
    callback to print loss every 100 epochs during AE training
    """

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 100 == 0:
            print(f"Training loss at epoch {epoch} is {logs.get('loss')}")


class Autoencoder(object):
    """
    Class that contains all the functions for AE training
    """

    def __init__(self, n_steps, lr, cu, activation, **kw):
        self._steps = n_steps
        self._alpha = lr
        self._activation = activation
        if 'nh1' in kw:
            self._h1 = self._h2 = kw['nh1']
        self._cu = cu
        if 'seed' in kw:
            self._seed = kw['seed']
        else:
            self._seed = 17


    def train_network(self, x_train, **kwargs):
        # tf.config.experimental_run_functions_eagerly(True)
        tf.compat.v1.disable_eager_execution()  # xps does not work with eager exec on. tf 2.1 bug?
        tf.keras.backend.clear_session()  # For easy reset of notebook state.
        tf.compat.v1.reset_default_graph()

        # to make this notebook's output stable across runs
        np.random.seed(self._seed)
        # tf.random.set_seed(self._seed) original
        #tf.random.set_random_seed(self._seed)

        # object for callback function during training
        loss_callback = LossCallback()

        # define model
        inputs = Input(shape=(len(x_train[0]),))
        hidden1 = Dense(self._h1, activation=self._activation)(inputs)
        hidden1 = Dense(self._h1, activation=self._activation)(hidden1)
        latent = Dense(self._cu)(hidden1)
        hidden2 = Dense(self._h2, activation=self._activation)(latent)
        hidden2 = Dense(self._h2, activation=self._activation)(hidden2)
        predictions = Dense(len(x_train[0]))(hidden2)

        if 'checkpoint' in kwargs:
            cp_callback = keras.callbacks.ModelCheckpoint(filepath=kwargs['checkpoint'] + 'model-{epoch:02d}.h5',
                                                          save_weights_only=True, verbose=0, period=2500)
        encoder = Model(inputs=inputs, outputs=latent)
        autoencoder = Model(inputs=inputs, outputs=predictions)

        autoencoder.summary()

        # compile model with mse loss and ADAM optimizer (uncomment for SGD)
        autoencoder.compile(loss='mse', optimizer=Adam(learning_rate=self._alpha))
        # autoencoder.compile(loss='mse', optimizer=SGD(learning_rate=self._alpha))

        # Specify path for TensorBoard log. Works only if typ is specified in kwargs
        if 'typ' in kwargs:
            log_dir = "logs\{}".format(kwargs['typ']) + "\{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if 'checkpoint' in kwargs:
            # Start training of the network
            history = autoencoder.fit(x=x_train,
                                      y=x_train,
                                      epochs=self._steps, verbose=0,
                                      batch_size=len(x_train),
                                      callbacks=[cp_callback, loss_callback])
        else:
            # Start training of the network
            history = autoencoder.fit(x=x_train,
                                      y=x_train,
                                      epochs=self._steps, verbose=0,
                                      batch_size=len(x_train),
                                      callbacks=[loss_callback])

        # Get network prediction
        # get_2nd_layer_output = K.function([autoencoder.layers[0].input],
        #                                   [autoencoder.layers[2].output])
        # train_cu = encoder.predict(x_train)
        # train_cu = get_2nd_layer_output([x_train])[0]
        train_cu = encoder.predict(x_train)
        train_rec = autoencoder.predict(x_train)

        weights = []
        biases = []
        # Get encoder parameters
        for layer in autoencoder.layers:
            if layer.get_weights():
                weights.append(layer.get_weights()[0])  # list of numpy arrays
                biases.append(layer.get_weights()[1])

        print("\n")  # blank space after loss printing

        # overload for different kwargs (test data, codings, ... )
        if 'x_test' in kwargs:
            test_rec = autoencoder.predict(kwargs['x_test'])
            test_cu = encoder.predict([kwargs['x_test']])
            return history, weights, biases, train_rec, train_cu, test_rec, test_cu
        else:
            return history, weights, biases, train_rec, train_cu

    def train_rnn(self, x_train, **kwargs):

        tf.keras.backend.clear_session()  # For easy reset of notebook state.
        tf.compat.v1.reset_default_graph()

        timesteps = 3
        n_features = x_train.shape[1]

        x_train = temporalize(X=x_train, lookback=timesteps)
        x_train = np.array(x_train)
        x_train = x_train.reshape(x_train.shape[0], timesteps, n_features)

        if 'x_test' in kwargs:
            x_test = temporalize(X=kwargs['x_test'], lookback=timesteps)

            x_test = np.array(x_test)
            x_test = x_test.reshape(x_test.shape[0], timesteps, n_features)

        # define model
        lstm_autoencoder = Sequential()
        lstm_autoencoder.add(LSTM(16, activation=self._activation, input_shape=(timesteps, n_features), return_sequences=False))
        lstm_autoencoder.add(Dense(2))
        # lstm_autoencoder.add(LSTM(2, activation=self._activation, return_sequences=False))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # lstm_autoencoder.add(LSTM(2, activation=self._activation, return_sequences=True))
        lstm_autoencoder.add(LSTM(16, activation=self._activation, return_sequences=True))
        lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

        lstm_autoencoder.summary()

        lstm_autoencoder.compile(optimizer='adam', loss='mse')

        # fit model
        lstm_autoencoder_history = lstm_autoencoder.fit(x_train, x_train, epochs=20, verbose=2)

        get_2nd_layer_output = K.function([lstm_autoencoder.layers[0].input],
                                          [lstm_autoencoder.layers[1].output])
        train_cu = get_2nd_layer_output([x_train])[0]

        # predict input signal
        train_rec = lstm_autoencoder.predict(x_train)
        train_rec_list = []
        for i in range(len(train_rec)):
            train_rec_list.append(train_rec[i][-1])
        train_rec = np.array(train_rec_list)
        train_rec = train_rec.reshape(train_rec.shape[0], n_features)

        # overload for different kwargs (test data, ... )
        if 'x_test' in kwargs:
            test_rec = lstm_autoencoder.predict(x_test)
            test_rec_list = []
            for i in range(len(test_rec)):
                test_rec_list.append(test_rec[i][-1])
            test_rec = np.array(test_rec_list)
            test_rec = test_rec.reshape(test_rec.shape[0], n_features)

            test_cu = get_2nd_layer_output([x_test])[0]
            return lstm_autoencoder_history, train_rec, train_cu, test_rec, test_cu
        else:
            return lstm_autoencoder_history, train_rec, train_cu

    def train_cnn(self, x_train, **kwargs):
        tf.keras.backend.clear_session()  # For easy reset of notebook state.
        tf.compat.v1.reset_default_graph()

        # to make this notebook's output stable across runs
        np.random.seed(self._seed)
        tf.random.set_seed(self._seed)

        # reshape input into a 4D tensor to perform convolutions (similar to LSTM)
        timesteps = 48
        n_features = x_train.shape[1]

        x_train = temporalize(X=x_train, lookback=timesteps)
        x_train = np.array(x_train)
        x_train = x_train.reshape(x_train.shape[0], timesteps, n_features, 1)

        if 'x_test' in kwargs:
            x_test = temporalize(X=kwargs['x_test'], lookback=timesteps)

            x_test = np.array(x_test)
            x_test = x_test.reshape(x_test.shape[0], timesteps, n_features, 1)

        # define model
        cnn_autoencoder = keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(timesteps, n_features, 1)),
            Conv2D(4, kernel_size=3, padding="SAME", activation=self._activation),
            Flatten(),
            Dense(2),
            Dense(timesteps*n_features*4, activation=self._activation),
            Reshape(target_shape=(timesteps, n_features, 4)),
            Conv2DTranspose(4, kernel_size=3, padding="SAME", activation=self._activation),
            Conv2DTranspose(1, kernel_size=3, padding="SAME"),
        ])
        cnn_autoencoder.summary()
        cnn_autoencoder.compile(optimizer='adam', loss='mse')

        # fit model
        cnn_autoencoder_history = cnn_autoencoder.fit(x_train, x_train, epochs=20, verbose=2)

        get_2nd_layer_output = K.function([cnn_autoencoder.layers[0].input],
                                          [cnn_autoencoder.layers[2].output])
        train_cu = get_2nd_layer_output([x_train])[0]

        # predict input signal
        train_rec = cnn_autoencoder.predict(x_train)
        train_rec_list = []
        for i in range(len(train_rec)):
            train_rec_list.append(train_rec[i][-1])
        train_rec = np.array(train_rec_list)
        train_rec = train_rec.reshape(train_rec.shape[0], n_features)

        # overload for different kwargs (test data, ... )
        if 'x_test' in kwargs:
            test_rec = cnn_autoencoder.predict(x_test)
            test_rec_list = []
            for i in range(len(test_rec)):
                test_rec_list.append(test_rec[i][-1])
            test_rec = np.array(test_rec_list)
            test_rec = test_rec.reshape(test_rec.shape[0], n_features)

            test_cu = get_2nd_layer_output([x_test])[0]
            return cnn_autoencoder_history, train_rec, train_cu, test_rec, test_cu
        else:
            return cnn_autoencoder_history, train_rec, train_cu

    def train_vae(self, x_train, **kwargs):

        # tf.config.experimental_run_functions_eagerly(True)
        tf.compat.v1.disable_eager_execution()  # xps does not work with eager exec on. tf 2.1 bug?
        tf.keras.backend.clear_session()  # For easy reset of notebook state.
        tf.compat.v1.reset_default_graph()

        # to make this notebook's output stable across runs
        np.random.seed(self._seed)
        # tf.random.set_seed(self._seed)

        # factor for scaling KLD term
        if 'beta' in kwargs:
            beta = kwargs['beta']
        else:
            beta = 0.001

        # object for callback function during training
        loss_callback = LossCallback()

        # Encoder definition
        x = Input(shape=len(x_train[0],), name="input")
        h = Dense(self._h1, activation=self._activation, name="intermediate_encoder")(x)
        h = Dense(self._h1, activation=self._activation, name="latent_encoder")(h)
        z_mean = Dense(self._cu, name="mu_encoder")(h)
        z_log_sigma = Dense(self._cu, name="sigma_encoder")(h)

        # Sampling trick from latent space
        def sampling(args):
            z_mean, z_log_sigma = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            std_dev = 0.1
            epsilon = K.random_normal(shape=(batch, dim),
                                      mean=0., stddev=std_dev)
            return z_mean + K.exp(z_log_sigma / 2.0) * epsilon

        z = Lambda(sampling)([z_mean, z_log_sigma])

        # Create encoder
        encoder = keras.Model(x, [z_mean, z_log_sigma, z], name='encoder')

        encoder.summary()

        # Decoder definition

        # Decoder definition
        decoder = Sequential([
            Dense(self._h2, input_dim=self._cu, activation=self._activation, name="input_decoder"),
            Dense(self._h2, input_dim=self._h2, activation=self._activation, name="intermediate_decoder"),
            Dense(len(x_train[0]), activation=self._activation, name="reconstruction_decoder")
        ])
        decoder.summary()

        # VAE model statement
        # instantiate VAE model
        pred = decoder(encoder(x)[2])
        vae = keras.Model(x, pred, name='vae_mlp')

        vae.summary()

        def vae_loss(input, output):
            # Compute error in reconstruction
            reconstruction_loss = mse(input, output)

            # Compute the KL Divergence regularization term
            kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)

            # Return the average loss over all images in batch
            total_loss = (reconstruction_loss + 0.0001 * kl_loss)
            return total_loss

        vae.compile(optimizer=Adam(learning_rate=self._alpha), loss=vae_loss)
        encoder.compile(optimizer=Adam(learning_rate=self._alpha), loss=vae_loss)
        decoder.compile(optimizer=Adam(learning_rate=self._alpha), loss=vae_loss)
        #vae.compile(optimizer=Adam(learning_rate=self._alpha))

        # It does not matter the type, but show us the vae

        # Harvard
        history = vae.fit(x=x_train,  y=x_train,
                shuffle=True,
                epochs=self._steps, verbose=0,
                batch_size=8, callbacks=[loss_callback])
                #validation_data=(val_x, None)

        # Get network prediction
        train_cu = encoder.predict(x_train)
        # do not sample from any distribution, just use the mean vector
        train_rec = vae.predict(x_train)

        weights = []
        biases = []
        # Get encoder/decoder parameters
        for layer in encoder.layers:
            if layer.get_weights():
                weights.append(layer.get_weights()[0])  # list of numpy arrays
                biases.append(layer.get_weights()[1])
        for layer in decoder.layers:
            if layer.get_weights():
                weights.append(layer.get_weights()[0])  # list of numpy arrays
                biases.append(layer.get_weights()[1])

        # after training it is time to generate some test signal. We start by sampling a set of latent vector from the
        # unit Gaussian distribution p(z). The generator will then convert the latent sample z to logits of the
        # observation, giving a distribution p(x/z).
        # Here the test set is not extracted by the distribution
        if 'x_test' in kwargs:
            test_rec = vae.predict(kwargs['x_test'])
            test_cu = encoder.predict(kwargs['x_test'])

            return history, weights, biases, train_rec, train_cu[2], test_rec, test_cu[2]
        else:
            return history, weights, biases, train_rec, train_cu[2]


class PrincipalComponentAnalysis(object):
    """
    Class that contains all the functions for PCA training
    """

    def __init__(self, n_PCs):
        self._pc = n_PCs

    def train_pca(self, train_signal, **kwargs):
        pca = PCA(n_components=len(train_signal[0]))
        pca.fit(train_signal)
        coeff = pca.components_.T

        train_score = np.matmul((train_signal - np.mean(train_signal, 0)), coeff)
        train_score[:, self._pc:] = 0
        train_score_out = train_score[:, 0:self._pc]
        train_signal_rec = np.matmul(train_score, coeff.T) + np.mean(train_signal, 0)

        if 'x_test' in kwargs:
            test_score = np.matmul((kwargs['x_test'] - np.mean(train_signal, 0)), coeff)
            test_score[:, self._pc:] = 0
            test_score_out = test_score[:, 0:self._pc]
            test_signal_rec = np.matmul(test_score, coeff.T) + np.mean(train_signal, 0)

            return pca, train_signal_rec, train_score_out, test_signal_rec, test_score_out
        else:
            return pca, train_signal_rec, train_score_out



def train_pca(calibPath, drPath, n_pc):
    """
    function to train BoMI forward map - PCA
    :param drPath: path to save BoMI forward map
    :return:
    """
    r = Reaching()
    # read calibration file and remove all the initial zero rows
    xp = list(pd.read_csv(calibPath + 'Calib.txt', sep=' ', header=None).values)
    x = [i for i in xp if all(i)]
    x = np.array(x)

    # randomly shuffle input
    np.random.shuffle(x)

    # define train/test split
    thr = 80
    split = int(len(x) * thr / 100)
    train_x = x[0:split, :]
    test_x = x[split:, :]

    # initialize object of class PCA
    PCA = PrincipalComponentAnalysis(n_pc)

    # train PCA
    pca, train_x_rec, train_pc, test_x_rec, test_pc = PCA.train_pca(train_x, x_test=test_x)
    print('PCA has been trained.')

    # save weights and biases
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    
    np.savetxt(drPath + "weights1.txt", pca.components_[:, :n_pc])

    print('BoMI forward map (PCA parameters) has been saved.')

    # compute train/test VAF
    print(f'Training VAF: {compute_vaf(train_x, train_x_rec)}')
    print(f'Test VAF: {compute_vaf(test_x, test_x_rec)}')

    # normalize latent space to fit the monitor coordinates
    # Applying rotation
    train_pc = np.dot(train_x, pca.components_[:, :n_pc])

    plot_map = False
    if plot_map:
        rot = 0
        train_pc_plot = reaching_functions.rotate_xy_RH(train_pc, rot)
        # Applying scale
        scale = [r.width / np.ptp(train_pc_plot[:, 0]), r.height / np.ptp(train_pc_plot[:, 1])]
        train_pc_plot = train_pc_plot * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_pc_plot[:, 0]), r.height / 2 - np.mean(train_pc_plot[:, 1])]
        train_pc_plot = train_pc_plot + off

        # Plot latent space
        plt.figure()
        plt.scatter(train_pc_plot[:, 0], train_pc_plot[:, 1], c='green', s=20)
        plt.title('Projections in workspace')
        plt.axis("equal")
        plt.show()
        
    print('You can continue with customization.')
    return train_pc


def train_ae(calibPath, drPath, n_map_component):
    """
    function to train BoMI forward map
    :param drPath: path to save BoMI forward map
    :return:
    """
    r = Reaching()

    # Autoencoder parameters
    n_steps = 3001
    lr = 0.02
    cu = n_map_component
    nh1 = 6
    activ = "tanh"

    # read calibration file and remove all the initial zero rows
    xp = list(pd.read_csv(calibPath + 'Calib.txt', sep=' ', header=None).values)
    x = [i for i in xp if all(i)]
    x = np.array(x)

    # randomly shuffle input
    np.random.shuffle(x)

    # define train/test split
    thr = 80
    split = int(len(x) * thr / 100)
    train_x = x[0:split, :]
    test_x = x[split:, :]

    # initialize object of class Autoencoder
    AE = Autoencoder(n_steps, lr, cu, activation=activ, nh1=nh1, seed=0)

    # train AE network
    history, ws, bs, train_x_rec, train_cu, test_x_rec, test_cu = AE.train_network(train_x, x_test=test_x)
    print('AE has been trained.')

    # save weights and biases
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    for layer in range(3):
        np.savetxt(drPath + "weights" + str(layer + 1) + ".txt", ws[layer])
        np.savetxt(drPath + "biases" + str(layer + 1) + ".txt", bs[layer])

    print('BoMI forward map (AE parameters) has been saved.')

    # compute train/test VAF
    print(f'Training VAF: {compute_vaf(train_x, train_x_rec)}')
    print(f'Test VAF: {compute_vaf(test_x, test_x_rec)}')

    # normalize latent space to fit the monitor coordinates
    # Applying rotation
    plot_ae = False
    if plot_ae:
        rot = 0
        train_cu_plot = reaching_functions.rotate_xy_RH(train_cu, rot)
        # Applying scale
        scale = [r.width / np.ptp(train_cu_plot[:, 0]), r.height / np.ptp(train_cu_plot[:, 1])]
        train_cu_plot = train_cu_plot * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_cu_plot[:, 0]), r.height / 2 - np.mean(train_cu_plot[:, 1])]
        train_cu_plot = train_cu_plot + off

        # Plot latent space
        plt.figure()
        plt.scatter(train_cu_plot[:, 0], train_cu_plot[:, 1], c='green', s=20)
        plt.title('Projections in workspace')
        plt.axis("equal")
        plt.show()

    print('You can continue with customization.')
    return train_cu

# Variational autoencoder
def train_vae(calibPath, drPath, n_map_component):
    """
    function to train BoMI forward map
    :param drPath: path to save BoMI forward map
    :return:
    """
    r = Reaching()

    # Autoencoder parameters
    n_steps = 3001
    lr = 0.02
    cu = n_map_component
    nh1 = 6
    activ = "tanh"

    # read calibration file and remove all the initial zero rows
    xp = list(pd.read_csv(calibPath + 'Calib.txt', sep=' ', header=None).values)
    x = [i for i in xp if all(i)]
    x = np.array(x)

    # randomly shuffle input
    np.random.shuffle(x)

    # define train/test split
    thr = 80
    split = int(len(x) * thr / 100)
    train_x = x[0:split, :]
    test_x = x[split:, :]

    # initialize object of class Autoencoder
    AE = Autoencoder(n_steps, lr, cu, activation=activ, nh1=nh1, seed=0)

    # train VAE network
    history, ws, bs, train_x_rec, train_cu, test_x_rec, test_cu = AE.train_vae(train_x, beta=0.00035, x_test=test_x)
    print('VAE has been trained.')

    # save weights and biases
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    for layer in range(5):
        np.savetxt(drPath + "weights" + str(layer + 1) + ".txt", ws[layer])
        np.savetxt(drPath + "biases" + str(layer + 1) + ".txt", bs[layer])

    print('BoMI forward map (VAE parameters) has been saved.')

    # compute train/test VAF
    print(f'Training VAF: {compute_vaf(train_x, train_x_rec)}')
    print(f'Test VAF: {compute_vaf(test_x, test_x_rec)}')

    # normalize latent space to fit the monitor coordinates
    # Applying rotation
    plot_vae = False
    if plot_vae:
        rot = 0
        train_cu_plot = reaching_functions.rotate_xy_RH(train_cu, rot)
        # Applying scale
        scale = [r.width / np.ptp(train_cu_plot[:, 0]), r.height / np.ptp(train_cu_plot[:, 1])]
        train_cu_plot = train_cu_plot * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_cu_plot[:, 0]), r.height / 2 - np.mean(train_cu_plot[:, 1])]
        train_cu_plot = train_cu_plot + off

        # Plot latent space
        plt.figure()
        plt.scatter(train_cu_plot[:, 0], train_cu_plot[:, 1], c='green', s=20)
        plt.title('Projections in workspace')
        plt.axis("equal")
        plt.show()

    print('You can continue with customization.')
    return train_cu

def load_bomi_map(dr_mode, drPath):
    map = None
    if dr_mode == 'pca':

        map = pd.read_csv(drPath + 'weights1.txt', sep=' ', header=None).values

    elif dr_mode == 'ae':
        ws = []
        bs = []
        ws.append(pd.read_csv(drPath + 'weights1.txt', sep=' ', header=None).values)
        ws.append(pd.read_csv(drPath + 'weights2.txt', sep=' ', header=None).values)
        ws.append(pd.read_csv(drPath + 'weights3.txt', sep=' ', header=None).values)
        bs.append(pd.read_csv(drPath + 'biases1.txt', sep=' ', header=None).values)
        bs[0] = bs[0].reshape((bs[0].size,))
        bs.append(pd.read_csv(drPath + 'biases2.txt', sep=' ', header=None).values)
        bs[1] = bs[1].reshape((bs[1].size,))
        bs.append(pd.read_csv(drPath + 'biases3.txt', sep=' ', header=None).values)
        bs[2] = bs[2].reshape((bs[2].size,))

        map = (ws, bs)

    elif dr_mode == 'vae':

        ws = []
        bs = []

        ws.append(pd.read_csv(drPath + 'weights1.txt', sep=' ', header=None).values)
        ws.append(pd.read_csv(drPath + 'weights2.txt', sep=' ', header=None).values)
        ws.append(pd.read_csv(drPath + 'weights3.txt', sep=' ', header=None).values)
        ws.append(pd.read_csv(drPath + 'weights4.txt', sep=' ', header=None).values)
        bs.append(pd.read_csv(drPath + 'biases1.txt', sep=' ', header=None).values)
        bs[0] = bs[0].reshape((bs[0].size,))
        bs.append(pd.read_csv(drPath + 'biases2.txt', sep=' ', header=None).values)
        bs[1] = bs[1].reshape((bs[1].size,))
        bs.append(pd.read_csv(drPath + 'biases3.txt', sep=' ', header=None).values)
        bs[2] = bs[2].reshape((bs[2].size,))
        bs.append(pd.read_csv(drPath + 'biases4.txt', sep=' ', header=None).values)
        bs[3] = bs[3].reshape((bs[3].size,))

        map = (ws, bs)

    return map

def save_bomi_map(q_body, drPath, r):

    body_calib = []

    keep_reading_q = True

    try:
        body_calib.append(q_body.get(block=True, timeout=10.0))
    except queue.Empty:
        print("WARN: no body data retrieved after 10 seconds. Is the detection working?")
        return

    while keep_reading_q:
        try:
            body_calib.append(q_body.get(block=True, timeout=1.0))
        except queue.Empty:
            # care only if the acquisition process ended
            if r.is_terminated:
                keep_reading_q = False

    body_calib = np.array(body_calib)
    if not os.path.exists(drPath):
        os.makedirs(drPath)
    np.savetxt(drPath + "Calib.txt", body_calib)

def read_transform(drPath, spec):
    rot = pd.read_csv(drPath + 'rotation_'+spec+'.txt', sep=' ', header=None).values
    scale = pd.read_csv(drPath + 'scale_'+spec+'.txt', sep=' ', header=None).values
    scale = np.reshape(scale, (scale.shape[0],))
    off = pd.read_csv(drPath + 'offset_'+spec+'.txt', sep=' ', header=None).values
    off = np.reshape(off, (off.shape[0],))
    return rot, scale, off

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


