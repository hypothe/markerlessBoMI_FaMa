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


def mse_loss(y_true, y_pred):
    """
    function to save MSE term in history when training VAE
    :param y_true: input signal
    :param y_pred: input signal predicted by the VAR
    :return: MSE
    """
    # E[log P(X|z)]. MSE loss term
    return K.mean(K.square(y_pred - y_true), axis=-1)


def kld_loss(codings_log_var, codings_mean, beta):
    """
    function to save KLD term in history when training VAE
    :param codings_log_var: log variance of AE codeunit
    :param codings_mean: mean of AE codeunit
    :param beta: scalar to weight KLD term
    :return: beta*KLD
    """

    def kld_loss(y_true, y_pred):
        # D_KL(Q(z|X) || P(z|X)); KLD loss term
        return beta * (-0.5 * K.sum(1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean), axis=-1))

    return kld_loss

# construct a custom layer to calculate the loss
class CustomVariationalLayer(Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.losses.binary_crossentropy(x, z_decoded)
        return xent_loss

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


def custom_loss_vae(codings_log_var, codings_mean, beta):
    """
    define cost function for VAE
    :param codings_log_var: log variance of AE codeunit
    :param codings_mean: mean of AE codeunit
    :param beta: scalar to weight KLD term
    :return: MSE + beta*KLD
    """

    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        mse_loss = K.mean(K.square(y_pred - y_true), axis=-1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kld_loss = -0.5 * K.sum(1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean), axis=-1)

        return mse_loss + beta*kld_loss

    return vae_loss


class Sampling(keras.layers.Layer):
    """
    Class to random a sample from gaussian distribution with given mean and std. Needed for reparametrization trick
    """

    def call(self, inputs):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
           # Arguments
               inputs (tensor): mean and log of variance of Q(z|X)
           # Returns
               z (tensor): sampled latent vector
           """
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon


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

    # def my_bias(shape, dtype=dtype):
    #     return K.random_normal(shape, dtype=dtype)

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

        # the inference network (encoder) defines an approximate posterior distribution q(z/x), which takes as input an
        # observation and outputs a set of parameters for the conditional distribution of the latent representation.
        # Here, I simply model this distribution as a diagional Gaussian. Specifically, the interfence network outputs
        # the mean and log-variance parameters of a factorized Gaussian (log-variance instead of the variance directly
        # is for numerical stability)

        # Decoder definition
        decoder = Sequential([
            Dense(self._h2, input_dim=self._cu, activation=self._activation, name="input_decoder"),
            Dense(self._h2, input_dim=self._h2, activation=self._activation, name="intermediate_decoder"),
            Dense(len(x_train[0]), activation=self._activation, name="reconstruction_decoder")
        ])
        decoder.summary()

        # Encoder definition
        x = Input(shape=len(x_train[0],), name="input")
        h = Dense(self._h1, activation=self._activation, name="intermediate_encoder")(x)
        h = Dense(self._h1, activation=self._activation, name="latent_decoder")(h)
        # During optimization, we can sample from q(z/x) by first sampling from a unit Gaussian, and then multiplying
        # by the standard deviation and adding the mean. This ensures the gradients could pass through the sample
        # to the interence network parameters. This is called reparametrization trick
        z_mu = Dense(self._cu)(h)
        z_log_var = Dense(self._cu)(h)

        # Harvard version ######

        # sampling function
        def sampling(args):
            z_mu, z_log_var = args
            epsilon_std = 1.0
            epsilon = K.random_normal(shape=(K.shape(z_mu)[0], self._cu),
                                      mean=0., stddev=epsilon_std)
            return z_mu + K.exp(z_log_var) * epsilon

        # sample vector from the latent distribution
        z = Lambda(sampling)([z_mu, z_log_var])
        x_pred = decoder(z)

        # apply the custom loss to the input images and the decoded latent distribution sample
        y = CustomVariationalLayer()([x, x_pred])

        # VAE model statement
        vae = Model(x, y)
        vae.compile(optimizer='rmsprop', loss=None)

        # Tiao Version #######

        # z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        # z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

        #epsilon_std = 1.0
        # eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], self._cu)))
        # z_eps = Multiply()([z_sigma, eps])
        # z = Add()([z_mu, z_eps])
        # vae = Model(inputs=[x, eps], outputs=x_pred)
        #  x_pred = decoder(z)
        #vae.compile(optimizer=Adam(learning_rate=self._alpha),
                        #       loss=custom_loss_vae(z_sigma, z_mu, beta),
                         #      metrics=[mse_loss, kld_loss(z_sigma, z_mu, beta)])



        #vae.compile(optimizer=Adam(learning_rate=self._alpha), loss=nll)

        # It does not matter the the type, but show us the vae
        vae.summary()

        # During training, 1. we start by iterating over the dataset
        # 2. during each iter, we pass the input data to the encoder to obtain a set of mean and log-variance
        # parameters of the approximate posterior q(z/x)
        # 3. we then apply the reparametrization trick to sample from q(z/x)
        # 4. finally, we pass the reparam samples to the decoder to obtain the logits of the generative distrib p(x/z)

        # Harvard
        history = vae.fit(x=x_train, y=None,
                shuffle=True,
                epochs=self._steps,
                batch_size=len(x_train))
                #validation_data=(val_x, None))

        # Tiao
        #history = vae.fit(x=x_train,
         #                            y=x_train,
          #                           epochs=self._steps, verbose=0,
           #                          batch_size=len(x_train),
            #                         callbacks=[loss_callback])


       # history = variational_ae.fit(x_train,
        #            x_train,
         #           shuffle=True,
          #          epochs=self._steps,
           #         batch_size=len(x_train)) #,
                    #validation_data=(x_test, x_test))

        encoder = Model(x, z_mu)

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
        if 'x_test' in kwargs:
            test_rec = vae.predict(kwargs['x_test'])
            test_cu = encoder.predict(kwargs['x_test'])

            return history, weights, biases, train_rec, train_cu, test_rec, test_cu
        else:
            return history, weights, biases, train_rec, train_cu


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
        train_pc = reaching_functions.rotate_xy_RH(train_pc, rot)
        # Applying scale
        scale = [r.width / np.ptp(train_pc[:, 0]), r.height / np.ptp(train_pc[:, 1])]
        train_pc = train_pc * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_pc[:, 0]), r.height / 2 - np.mean(train_pc[:, 1])]
        train_pc = train_pc + off

        # Plot latent space
        plt.figure()
        plt.scatter(train_pc[:, 0], train_pc[:, 1], c='green', s=20)
        plt.title('Projections in workspace')
        plt.axis("equal")
        plt.show()

        # save PCA scaling values
        with open(drPath + "rotation_dr.txt", 'w') as f:
            print(rot, file=f)
        np.savetxt(drPath + "scale_dr.txt", scale)
        np.savetxt(drPath + "offset_dr.txt", off)

        print('PCA scaling values has been saved.')
        
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
        train_cu = reaching_functions.rotate_xy_RH(train_cu, rot)
        #if cu == 3:
        #    train_cu[2] = np.tanh(train_cu[2])
        # Applying scale
        scale = [r.width / np.ptp(train_cu[:, 0]), r.height / np.ptp(train_cu[:, 1])]
        train_cu = train_cu * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_cu[:, 0]), r.height / 2 - np.mean(train_cu[:, 1])]
        train_cu = train_cu + off

        # Plot latent space
        plt.figure()
        plt.scatter(train_cu[:, 0], train_cu[:, 1], c='green', s=20)
        plt.title('Projections in workspace')
        plt.axis("equal")
        plt.show()

        # save AE scaling values
        with open(drPath + "rotation_dr.txt", 'w') as f:
            print(rot, file=f)
        np.savetxt(drPath + "scale_dr.txt", scale)
        np.savetxt(drPath + "offset_dr.txt", off)

        print('AE scaling values has been saved.')

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
    for layer in range(3):
        np.savetxt(drPath + "weights" + str(layer + 1) + ".txt", ws[layer])
        np.savetxt(drPath + "biases" + str(layer + 1) + ".txt", bs[layer])

    print('BoMI forward map (VAE parameters) has been saved.')

    # compute train/test VAF
    print(f'Training VAF: {compute_vaf(train_x, train_x_rec)}')
    print(f'Test VAF: {compute_vaf(test_x, test_x_rec)}')

    # normalize latent space to fit the monitor coordinates
    # Applying rotation
    plot_ae = False
    if plot_ae:
        rot = 0
        train_cu = reaching_functions.rotate_xy_RH(train_cu, rot)
        #if cu == 3:
        #    train_cu[2] = np.tanh(train_cu[2])
        # Applying scale
        scale = [r.width / np.ptp(train_cu[:, 0]), r.height / np.ptp(train_cu[:, 1])]
        train_cu = train_cu * scale
        # Applying offset
        off = [r.width / 2 - np.mean(train_cu[:, 0]), r.height / 2 - np.mean(train_cu[:, 1])]
        train_cu = train_cu + off

        # Plot latent space
        plt.figure()
        plt.scatter(train_cu[:, 0], train_cu[:, 1], c='green', s=20)
        plt.title('Projections in workspace')
        plt.axis("equal")
        plt.show()

        # save AE scaling values
        with open(drPath + "rotation_dr.txt", 'w') as f:
            print(rot, file=f)
        np.savetxt(drPath + "scale_dr.txt", scale)
        np.savetxt(drPath + "offset_dr.txt", off)

        print('VAE scaling values has been saved.')

    print('You can continue with customization.')
    return train_cu

def load_bomi_map(dr_mode, drPath):

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
        bs.append(pd.read_csv(drPath + 'biases1.txt', sep=' ', header=None).values)
        bs[0] = bs[0].reshape((bs[0].size,))
        bs.append(pd.read_csv(drPath + 'biases2.txt', sep=' ', header=None).values)
        bs[1] = bs[1].reshape((bs[1].size,))
        bs.append(pd.read_csv(drPath + 'biases3.txt', sep=' ', header=None).values)
        bs[2] = bs[2].reshape((bs[2].size,))

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


