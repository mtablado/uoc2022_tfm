import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PIXELS = 128
#EPOCHS = 500
EPOCHS = 1
print("WARNING: EPOCH is only 1!!!!!")

DATASET = '/dataset/'
MODELS = '/dataset/models/'

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def vae_encoder():
  latent_dim = 2

  inputs = keras.Input(shape=(PIXELS, PIXELS, 1), name='input_layer')

  # Block-1
  x = layers.Conv2D(16, kernel_size=4, strides= 2, padding='same', name='conv_1')(inputs)
  x = layers.BatchNormalization(name='bn_1')(x)
  x = layers.LeakyReLU(name='lrelu_1')(x)

  # Block-2
  x = layers.Conv2D(32, kernel_size=4, strides= 2, padding='same', name='conv_2')(x)
  x = layers.BatchNormalization(name='bn_2')(x)
  x = layers.LeakyReLU(name='lrelu_2')(x)

  # Block-3
  x = layers.Conv2D(64, 4, 2, padding='same', name='conv_3')(x)
  x = layers.BatchNormalization(name='bn_3')(x)
  x = layers.LeakyReLU(name='lrelu_3')(x)

  # Block-4
  x = layers.Conv2D(256, 4, 2, padding='same', name='conv_4')(x)
  x = layers.BatchNormalization(name='bn_4')(x)
  x = layers.LeakyReLU(name='lrelu_4')(x)

  # Block-5
  x = layers.Conv2D(1024, 4, 2, padding='same', name='conv_5')(x)
  x = layers.BatchNormalization(name='bn_5')(x)
  x = layers.LeakyReLU(name='lrelu_5')(x)

  # Final Block
  flatten = layers.Flatten()(x)
  mean = layers.Dense(1024, name='mean')(flatten)
  log_var = layers.Dense(1024, name='log_var')(flatten)
  z = Sampling()([mean, log_var])
  model = tf.keras.Model(inputs, (mean, log_var, z), name="Encoder")
  model.summary()
  return model

encoder = vae_encoder()

def vae_decoder():
  inputs = keras.Input(shape=(1024,), name='input_layer')
  x = layers.Dense(16384, name='dense_1')(inputs)
  x = layers.Reshape((4,4,1024), name='Reshape')(x)

  # Block-1
  x = layers.Conv2DTranspose(256, 4, strides= 2, padding='same',name='conv_transpose_1')(x)
  x = layers.BatchNormalization(name='bn_1')(x)
  x = layers.LeakyReLU(name='lrelu_1')(x)

  # Block-2
  x = layers.Conv2DTranspose(64, 4, strides= 2, padding='same', name='conv_transpose_2')(x)
  x = layers.BatchNormalization(name='bn_2')(x)
  x = layers.LeakyReLU(name='lrelu_2')(x)

  # Block-3
  x = layers.Conv2DTranspose(32, 4, 2, padding='same', name='conv_transpose_3')(x)
  x = layers.BatchNormalization(name='bn_3')(x)
  x = layers.LeakyReLU(name='lrelu_3')(x)

  # Block-4
  x = layers.Conv2DTranspose(16, 4, 2, padding='same', name='conv_transpose_4')(x)
  x = layers.BatchNormalization(name='bn_4')(x)
  x = layers.LeakyReLU(name='lrelu_4')(x)

  # Block-5
  outputs = layers.Conv2DTranspose(1, 4, 2,padding='same', activation='sigmoid', name='conv_transpose_5')(x)
  model = tf.keras.Model(inputs, outputs, name="Decoder")
  model.summary()
  return model

decoder = vae_decoder()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # The reconstruction loss measures how close the decoder output is to the original input by using the mean-squared error (MSE)
            r_loss = tf.reduce_mean(tf.square(data - reconstruction), axis = [1,2,3])
            reconstruction_loss = 1000 * r_loss

            #reconstruction_loss = tf.reduce_mean(
            #    tf.reduce_sum(
            #        keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #    )
            #)

            #reconstruction_loss = keras.metrics.mean_squared_error(data, reconstruction)

            # The KL loss, or Kullback–Leibler divergence, measures the difference between two probability distributions.
            # Minimizing the KL loss in this case means ensuring that the learned means and variances are as close as possible to those of the target (normal) distribution.
            # For a latent dimension of size K
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            #kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            #kl = tf.keras.losses.KLDivergence()
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl(data, reconstruction)))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        result = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        return result

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./(PIXELS - 1) # To normalize values
    )
train_generator = train_datagen.flow_from_directory(
        DATASET+'/slices-160-190/training',
        target_size=(PIXELS, PIXELS),
        batch_size=64, # Number of images processed together
        color_mode='grayscale',
        class_mode=None)

test_datagen = ImageDataGenerator(rescale=1./(PIXELS - 1))

test_generator = test_datagen.flow_from_directory(
        DATASET+'/slices-160-190/test',
        target_size=(PIXELS, PIXELS),
        batch_size=64,
        color_mode='grayscale',
        class_mode=None)

import timeit
vae = VAE(encoder, decoder)
vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005))

# Entrenar el modelo
t0 = timeit.default_timer()

mfit = vae.fit(train_generator, epochs=EPOCHS, batch_size=64)
training_time_ann = timeit.default_timer() - t0