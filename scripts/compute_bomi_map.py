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

        inputs = Input(shape=(len(x_train[0]),))
        hidden1 = Dense(self._h1, activation=self._activation)(inputs)
        hidden1 = Dense(self._h1, activation=self._activation)(hidden1)

        codings_mean = Dense(self._cu, name="encoder_mu")(hidden1)
        codings_log_var = Dense(self._cu, name="encoder_log_variance")(hidden1)
        # During optimization, we can sample from q(z/x) by first sampling from a unit Gaussian, and then multiplying
        # by the standard deviation and adding the mean. This ensures the gradients could pass through the sample
        # to the interence network parameters. This is called reparametrization trick
        # codings = Sampling()([codings_mean, codings_log_var])
        z = Sampling()([codings_mean, codings_log_var])

        variational_encoder = keras.Model(inputs, [codings_mean, codings_log_var, z], name="encoder")
        variational_encoder.summary()

        # the generative network (decoder)is just a mirrored version of the encoder.
        decoder_inputs = Input(shape=[self._cu])
        x = Dense(self._h1, activation=self._activation)(decoder_inputs)
        x = Dense(self._h1, activation=self._activation)(x)
        outputs = Dense(len(x_train[0]))(x)

        variational_decoder = Model(inputs=[decoder_inputs], outputs=[outputs])
        variational_decoder.summary()


        _, _, codings = variational_encoder(inputs)
        reconstructions = variational_decoder(codings)

        variational_ae = Model(inputs=inputs, outputs=reconstructions)

        variational_ae.compile(loss=custom_loss_vae(codings_log_var, codings_mean, beta),
                               optimizer=Adam(learning_rate=self._alpha),
                               metrics=[mse_loss, kld_loss(codings_log_var, codings_mean, beta)])


        # During training, 1. we start by iterating over the dataset
        # 2. during each iter, we pass the input data to the encoder to obtain a set of mean and log-variance
        # parameters of the approximate posterior q(z/x)
        # 3. we then apply the reparametrization trick to sample from q(z/x)
        # 4. finally, we pass the reparam samples to the decoder to obtain the logits of the generative distrib p(x/z)
        history = variational_ae.fit(x=x_train,
                                     y=x_train,
                                     epochs=self._steps, verbose=0,
                                     batch_size=len(x_train),
                                     callbacks=[loss_callback])

        # Get network prediction
        train_cu = variational_encoder.predict(x_train)
        # do not sample from any distribution, just use the mean vector
        train_rec = variational_decoder.predict(train_cu[0])
        # train_rec = variational_ae.predict(x_train)

        weights = []
        biases = []
        # Get encoder/decoder parameters
        for layer in variational_encoder.layers:
            if layer.get_weights():
                weights.append(layer.get_weights()[0])  # list of numpy arrays
                biases.append(layer.get_weights()[1])
        for layer in variational_decoder.layers:
            if layer.get_weights():
                weights.append(layer.get_weights()[0])  # list of numpy arrays
                biases.append(layer.get_weights()[1])

        # after training it is time to generate some test signal. We start by sampling a set of latent vector from the
        # unit Gaussian distribution p(z). The generator will then convert the latent sample z to logits of the
        # observation, giving a distribution p(x/z).
        if 'x_test' in kwargs:
            test_cu = variational_encoder.predict(kwargs['x_test'])
            test_rec = variational_decoder.predict(test_cu[0])

            return history, weights, biases, train_rec, train_cu, test_rec, test_cu
        else:
            return history, weights, biases, train_rec, train_cu

    # def train_adversarial(self, x_train, struct, **kwargs):
    #     """
    #     Deterministic unsupervised adversarial autoencoder.
    #     We are using:
    #         - Gaussian distribution as prior distribution.
    #         - Dense layers.
    #         - Cyclic learning rate.
    #     :param x_train:
    #     :param kwargs:
    #     :return:
    #     """
    #
    #     # tf.config.experimental_run_functions_eagerly(True)
    #     # tf.compat.v1.disable_eager_execution()  # xps does not work with eager exec on. tf 2.1 bug?
    #     tf.compat.v1.enable_eager_execution()
    #     tf.keras.backend.clear_session()  # For easy reset of notebook state.
    #     tf.compat.v1.reset_default_graph()
    #
    #     # to make this notebook's output stable across runs
    #     np.random.seed(self._seed)
    #     tf.random.set_seed(self._seed)
    #
    #     if 'x_test' in kwargs:
    #         x_test = kwargs['x_test']
    #     else:
    #         x_test = x_train
    #
    #     # define gaussian mixture model, if specified in the function argument
    #     if 'mvg' in kwargs:
    #         target_d = tfp.distributions.MultivariateNormalFullCovariance(loc=kwargs['param_d'][0],
    #                                                                       covariance_matrix=kwargs['param_d'][1])
    #     elif 'gmm' in kwargs:
    #         mix = 0.5
    #         target_d = tfp.distributions.Mixture(
    #             cat=tfp.distributions.Categorical(probs=[mix, 1. - mix]),
    #             components=[
    #                 tfp.distributions.Normal(loc=kwargs['param_d'][0], scale=kwargs['param_d'][1]),
    #                 tfp.distributions.Normal(loc=kwargs['param_d'][2], scale=kwargs['param_d'][3]),
    #             ])
    #
    #     # set number of latent units
    #     z_dim = self._cu
    #
    #     # prepare dictionary oh history
    #     history_adv = dict()
    #     history_adv['mse'] = []
    #     history_adv['dc_loss'] = []
    #     history_adv['dc_acc'] = []
    #     history_adv['gn_loss'] = []
    #
    #     # define model
    #     if struct == "conv":
    #         timesteps = 48
    #         n_features = x_train.shape[1]
    #
    #         x_train = temporalize(X=x_train, lookback=timesteps)
    #         x_train = np.array(x_train)
    #         x_train = x_train.reshape(x_train.shape[0], timesteps, n_features, 1)
    #
    #         if 'x_test' in kwargs:
    #             x_test = temporalize(X=kwargs['x_test'], lookback=timesteps)
    #
    #             x_test = np.array(x_test)
    #             x_test = x_test.reshape(x_test.shape[0], timesteps, n_features, 1)
    #
    #         # encoder model (this equals the generator of a GAN)
    #         inputs = Input(shape=(timesteps, n_features, 1))
    #         layer1 = Conv2D(4, kernel_size=3, padding="SAME", activation=self._activation)(inputs)
    #         layer2 = Flatten()(layer1)
    #         latent = Dense(2)(layer2)
    #         encoder = tf.keras.Model(inputs=inputs, outputs=latent)
    #
    #         # decoder model ((p(x/z)) to get back the original input space)
    #         encoded = Input(shape=(z_dim,))
    #         layer3 = Dense(timesteps * n_features * 4, activation=self._activation)(encoded)
    #         layer3 = keras.layers.Reshape(target_shape=(timesteps, n_features, 4))(layer3)
    #         layer4 = Conv2DTranspose(4, kernel_size=3, padding="SAME", activation=self._activation)(layer3)
    #         prediction = Conv2DTranspose(1, kernel_size=3, padding="SAME")(layer4)
    #         decoder = tf.keras.Model(inputs=encoded, outputs=prediction)
    #     else:
    #         # encoder model (this equals the generator of a GAN)
    #         inputs = Input(shape=(len(x_train[0]),))
    #         hidden = Dense(self._h1, activation=self._activation)(inputs)
    #         hidden = Dense(self._h1, activation=self._activation)(hidden)
    #         latent = Dense(z_dim)(hidden)
    #         encoder = tf.keras.Model(inputs=inputs, outputs=latent)
    #
    #         # decoder model ((p(x/z)) to get back the original input space)
    #         encoded = Input(shape=(z_dim,))
    #         hidden_d = Dense(self._h2, activation=self._activation)(encoded)
    #         hidden_d = Dense(self._h2, activation=self._activation)(hidden_d)
    #         prediction = Dense(len(x_train[0]))(hidden_d)
    #         decoder = tf.keras.Model(inputs=encoded, outputs=prediction)
    #
    #     # discriminator model (to tell if the samples of the latent space are from a prior distribution (p(z))
    #     # or from the output of the encoder (z)
    #     encoded_discriminator = Input(shape=(z_dim,))
    #     layer3_discriminator = Dense(self._h2, activation=self._activation)(encoded_discriminator)
    #     layer3_discriminator = Dense(self._h2, activation=self._activation)(layer3_discriminator)
    #     prediction_discriminator = tf.keras.layers.Dense(1, activation='sigmoid')(layer3_discriminator)
    #     # prediction_discriminator = tf.keras.layers.Dense(1)(layer3_discriminator)
    #     discriminator = tf.keras.Model(inputs=encoded_discriminator, outputs=prediction_discriminator)
    #
    #     # summary
    #     encoder.summary()
    #     decoder.summary()
    #     discriminator.summary()
    #
    #     # Define loss functions
    #     ae_loss_weight = 1
    #     dc_loss_weight = 1
    #     gen_loss_weight = 2
    #
    #     # Computes the cross-entropy loss between true labels and predicted labels.
    #     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #     mse = tf.keras.losses.MeanSquaredError()
    #     accuracy = tf.keras.metrics.BinaryAccuracy()
    #
    #     # I need three cost functions: training an AAE has two parts.
    #     # First being the reconstruction phase (we’ll train our autoencoder to reconstruct the input (i))
    #     # Second being the regularization phase (first the discriminator (ii) is trained followed by the encoder (iii)).
    #
    #     # this term of the loss is the usual mse between inputs and outputs.
    #     def autoencoder_loss(inputs, reconstruction, loss_weight):
    #         return loss_weight * mse(inputs, reconstruction)
    #
    #     # these next two loss terms serve to train the discriminator and the generator (details later).
    #     # disriminator should give us an output 1 if we pass in random inputs with desired distribution (real output)
    #     # disriminator should give us an output 0 (fake output) when we pass in the encoder output
    #     def discriminator_loss(real_output, fake_output, loss_weight):
    #         loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    #         loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #         return loss_weight * (loss_fake + loss_real)
    #
    #     # loss between fake output (0 or 1, encoder+decoder) and 1. Target is fixed to 1 (at the discriminator output)
    #     # to force the generator (encoder) generate samples whose latent space has specific prior distribution
    #     def generator_loss(real_output, fake_output, loss_weight):
    #         loss_real = cross_entropy(tf.ones_like(fake_output), fake_output)
    #         loss_fake = cross_entropy(tf.zeros_like(real_output), real_output)
    #         # return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)
    #         return loss_weight * (loss_fake + loss_real)
    #
    #     # autoencoder_loss + generator together! Learning rate of genertator and discrim different!
    #
    #
    #     # Define cyclic learning rate
    #     # base_lr = 0.001
    #     # max_lr = 0.02
    #     # -------------------------------------------------------------------------------------------------------------
    #     # Define optimizers
    #     ae_optimizer = tf.keras.optimizers.Adam(learning_rate=self._alpha)
    #     dc_optimizer = tf.keras.optimizers.Adam(learning_rate=self._alpha/3)
    #     gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self._alpha)
    #
    #     # # define vector of random guassian values to be learnt by the model.
    #     # # Do I need to define it once at the start or am I allowed to generate a new vector at each step of the train?
    #     # real_distribution = tf.random.normal([x_train.shape[0], z_dim], mean=0.0, stddev=1.0)
    #
    #     # -------------------------------------------------------------------------------------------------------------
    #     # Training function
    #     @tf.function
    #     def train_step(batch_x):
    #         # -------------------------------------------------------------------------------------------------------------
    #         # Autoencoder. This is the reconstruction phase. We’ll train both the encoder and the decoder to minimize
    #         # the reconstruction loss (mean squared error between the input and the decoder output).
    #         # Forget that the discriminator even exists in this phase. As usual we’ll pass inputs to the encoder which
    #         # will give us our latent code, later, we’ll pass this latent code to the decoder to get back the input.
    #         # We’ll backprop through both the encoder and the decoder weights so that rec loss will be reduced.
    #         #
    #         # NOTE: with TF 2.0 eager execution is enabled. Thus, TF will calculate the values of tensors as they
    #         # occur in the code. This means that it won't precompute a static graph for which inputs are fed in through
    #         # placeholders. This means that to back propagate errors, I have to keep track of the gradients of the
    #         # computation and then apply these gradients to an optimizer. This is what GradientTape does.
    #         # Because tensors are evaluated immediately, you don't have a graph to calculate gradients
    #         # and so you do need a gradient tape
    #         with tf.GradientTape() as ae_tape:
    #             encoder_output = encoder(batch_x)
    #             decoder_output = decoder(encoder_output)
    #             encoder.trainable = True
    #             decoder.trainable = True
    #             discriminator.trainable = False
    #
    #             # Autoencoder loss
    #             ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)
    #
    #         ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    #         ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))
    #
    #         # -------------------------------------------------------------------------------------------------------------
    #         # Discriminator. This is the first step of the regularization phase. In this phase I have to train the
    #         # discriminator and the generator (which is nothing but the encoder). Just forget that the decoder exists.
    #         # First, train the discriminator to classify the encoder output (z) and some random input
    #         # (z’, with the required distribution). The discriminator should give an output of 1 if I pass
    #         # in random inputs with the desired distribution (real values) and should give an output 0 (fake values)
    #         # when I pass in the encoder output.
    #         # Intuitively, both the encoder output and the random inputs to the discriminator should have the same size.
    #         # SUMMARY: I first train the discriminator to distinguish between the real distribution samples and the
    #         # fake ones from the generator (encoder in this case). So basically, you train the discriminator such that
    #         # when the input is encoder output it gives 0, and when the input is real distrib it gives 1
    #         # real distribution to be one
    #         with tf.GradientTape() as dc_tape:
    #             if 'mvg' in kwargs:
    #                 real_distribution = target_d.sample(sample_shape=(batch_x.shape[0], ), )
    #             if 'gmm' in kwargs:
    #                 real_distribution = target_d.sample(sample_shape=(batch_x.shape[0], z_dim), )
    #             else:
    #                 # real_distribution = tf.random.uniform(shape=[batch_x.shape[0], z_dim], minval=-1., maxval=1.)
    #                 real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
    #
    #             encoder_output = encoder(batch_x)
    #             encoder.trainable = False
    #             decoder.trainable = False
    #             discriminator.trainable = True
    #
    #             # only weights not fixed
    #             dc_real = discriminator(real_distribution)
    #             # Here training=False since we want the same discriminator weights in the second call
    #             # (if this is not specified, then tensorflow creates new set of randomly initialized weights.
    #             dc_fake = discriminator(encoder_output)
    #
    #             # Discriminator Loss
    #             dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)
    #
    #             # Discriminator Acc
    #             dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
    #                               tf.concat([dc_real, dc_fake], axis=0))
    #
    #         dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    #         dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))
    #
    #         # -------------------------------------------------------------------------------------------------------------
    #         # Generator (Encoder). This is second step of the regularization phase. Here I force the encoder to output
    #         # latent code with the desired distribution. To accomplish this I connect the encoder output as the
    #         # input to the discriminator. I fix the discriminator weights to whatever they are currently
    #         # (make them untrainable) and fix the target to 1 at the discriminator output. Later, we pass in inputs to
    #         # the encoder and find the discriminator output which is then used to find the loss (cross-entropy
    #         # cost function). I backprop only through the encoder weights, which causes the encoder to learn the
    #         # required distribution and produce output which’ll have that distribution (fixing the discriminator target
    #         # to 1 should cause the encoder to learn the required distribution by looking at the discriminator weights).
    #         # SUMMARY: Next step will be to train the generator (encoder) to output a required distribution.
    #         # This requires the discrminator’s target to be set to 1 (done in the generator loss)
    #         # and the dc_fake variable (the encoder connected to the discriminator). To update only the required weights
    #         # during training we’ll need to pass in all those collected weights to the var_list parameter.
    #         # So, I’ve passed in the discriminator variables (dc_var) and the generator (encoder) variables (en_var)
    #         # during their training phases. If the generator loss decreases, it means that the fake values (produced by
    #         # the encoder-generator) are close to the real values from the target distribution. So basically, you train
    #         # the generator such that the output of the discriminator is 1
    #         with tf.GradientTape() as gen_tape:
    #             if 'mvg' in kwargs:
    #                 real_distribution = target_d.sample(sample_shape=(batch_x.shape[0], ), )
    #             if 'gmm_param' in kwargs:
    #                 real_distribution = target_d.sample(sample_shape=(batch_x.shape[0], z_dim))
    #             else:
    #                 # real_distribution = tf.random.uniform(shape=[batch_x.shape[0], z_dim], minval=-1., maxval=1.)
    #                 real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
    #             encoder_output = encoder(batch_x)
    #             dc_real = discriminator(real_distribution)
    #             dc_fake = discriminator(encoder_output)
    #             encoder.trainable = True
    #             decoder.trainable = False
    #             discriminator.trainable = False
    #
    #             # Generator loss
    #             gen_loss = generator_loss(dc_real, dc_fake, gen_loss_weight)
    #
    #         gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    #         gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))
    #
    #         return ae_loss, dc_loss, dc_acc, gen_loss
    #
    #     # -------------------------------------------------------------------------------------------------------------
    #     # Training loop
    #     n_epochs = self._steps
    #     for epoch in range(n_epochs):
    #
    #         # Values of loss are saved as Tensor. I have to run them in a sess to get a value
    #         epoch_ae_loss_avg = tf.metrics.Mean()
    #         epoch_dc_loss_avg = tf.metrics.Mean()
    #         epoch_dc_acc_avg = tf.metrics.Mean()
    #         epoch_gen_loss_avg = tf.metrics.Mean()
    #
    #         # -------------------------------------------------------------------------------------------------------------
    #         ae_loss, dc_loss, dc_acc, gen_loss = train_step(x_train)
    #
    #         epoch_ae_loss_avg(ae_loss)
    #         epoch_dc_loss_avg(dc_loss)
    #         epoch_dc_acc_avg(dc_acc)
    #         epoch_gen_loss_avg(gen_loss)
    #
    #         history_adv['mse'].append(epoch_ae_loss_avg.result())
    #         history_adv['dc_loss'].append(epoch_dc_loss_avg.result())
    #         history_adv['dc_acc'].append(epoch_dc_acc_avg.result())
    #         history_adv['gn_loss'].append(epoch_gen_loss_avg.result())
    #
    #         # -------------------------------------------------------------------------------------------------------------
    #         if epoch % 100 == 0:
    #             print('{:8d}: AE_LOSS: {:.8f} DC_LOSS: {:.8f} DC_ACC: {:.8f} GEN_LOSS: {:.8f}' \
    #                   .format(epoch,
    #                           epoch_ae_loss_avg.result(),
    #                           epoch_dc_loss_avg.result(),
    #                           epoch_dc_acc_avg.result(),
    #                           epoch_gen_loss_avg.result()))
    #
    #     # Get network prediction
    #     train_cu = np.array(encoder(x_train, training=False))
    #     train_rec = np.array(decoder(encoder(x_train, training=False), training=False))
    #
    #     if struct == "conv":
    #         train_rec_list = []
    #         for i in range(len(train_rec)):
    #             train_rec_list.append(train_rec[i][-1])
    #         train_rec = np.array(train_rec_list)
    #         train_rec = train_rec.reshape(train_rec.shape[0], n_features)
    #
    #     weights = []
    #     biases = []
    #     # Get encoder/decoder parameters
    #     for layer in encoder.layers:
    #         if layer.get_weights():
    #             weights.append(layer.get_weights()[0])  # list of numpy arrays
    #             biases.append(layer.get_weights()[1])
    #     for layer in decoder.layers:
    #         if layer.get_weights():
    #             weights.append(layer.get_weights()[0])  # list of numpy arrays
    #             biases.append(layer.get_weights()[1])
    #
    #     if 'x_test' in kwargs:
    #         test_cu = np.array(encoder(x_test, training=False))
    #         test_rec = np.array(decoder(encoder(x_test, training=False), training=False))
    #
    #         if struct == "conv":
    #             test_rec_list = []
    #             for i in range(len(test_rec)):
    #                 test_rec_list.append(test_rec[i][-1])
    #             test_rec = np.array(test_rec_list)
    #             test_rec = test_rec.reshape(test_rec.shape[0], n_features)
    #
    #         return history_adv, weights, biases, train_rec, train_cu, test_rec, test_cu
    #     else:
    #         return history_adv, weights, biases, train_rec, train_cu


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

    elif dr_mode == 'vae':

        ws = []
        bs = []

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

