import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.datasets import mnist


def build_cnn_vae(input_shape, latent_size=5):
    # initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.0000001)
    initializer = tf.keras.initializers.Zeros()

    image_size = input_shape[1]
    assert image_size % 4 == 0

    input_ = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(input_)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    mean = tf.keras.layers.Dense(latent_size, kernel_initializer=initializer)(x)
    log_std = tf.keras.layers.Dense(latent_size, kernel_initializer=initializer)(x)
    latent_vector = tf.keras.layers.Lambda(sample)([mean, log_std])
    encoder = tf.keras.Model(inputs=input_, outputs=[mean, log_std, latent_vector])

    print(encoder.summary())

    dim = image_size // 4

    latent_inputs = tf.keras.Input(shape=(latent_size,))
    x = tf.keras.layers.Dense(units=dim * dim * 32, activation=tf.nn.relu)(latent_inputs)
    x = tf.keras.layers.Reshape(target_shape=(dim, dim, 32))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    output_ = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=output_)

    print(decoder.summary())

    vae = decoder(encoder(input_)[2])
    vae_model = tf.keras.Model(inputs=input_, outputs=vae)

    def vae_loss(inputs, outputs):
        reconstruction_loss = K.sum(K.square(outputs - inputs))
        # print(type(reconstruction_loss))
        kl_loss = -0.5 * K.sum((1 + log_std - K.square(mean) - K.square(K.exp(log_std))), axis=-1)
        # print(type(kl_loss))
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss

    return encoder, decoder, vae_model, vae_loss


def build_vae(input_shape, latent_size=5, nodes=(1000, 500, 120)):
    # initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.001)
    initializer = tf.keras.initializers.Zeros()

    input_ = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_)
    for node in nodes:
        x = tf.keras.layers.Dense(node, activation="relu")(x)

    mean = tf.keras.layers.Dense(latent_size, kernel_initializer=initializer)(x)
    log_std = tf.keras.layers.Dense(latent_size, kernel_initializer=initializer)(x)
    latent_vector = tf.keras.layers.Lambda(sample)([mean, log_std])
    encoder = tf.keras.Model(inputs=input_, outputs=[mean, log_std, latent_vector])

    latent_inputs = tf.keras.Input(shape=(latent_size,))
    x = latent_inputs
    for node in reversed(nodes):
        x = tf.keras.layers.Dense(node, activation="relu")(x)

    output_ = tf.keras.layers.Dense(np.prod(input_shape))(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=output_)

    vae = decoder(encoder(input_)[2])
    vae_model = tf.keras.Model(inputs=input_, outputs=vae)

    def vae_loss(inputs, outputs):
        reconstruction_loss = K.sum(K.square(outputs - inputs))
        # print(type(reconstruction_loss))
        kl_loss = -0.5 * K.sum((1 + log_std - K.square(mean) - K.square(K.exp(log_std))), axis=-1)
        # print(type(kl_loss))
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss

    return encoder, decoder, vae_model, vae_loss


def sample(args):
    mean, log_std = args
    norm_sample = tf.random.normal(tf.shape(mean), mean=mean, stddev=K.square(K.exp(log_std)))
    return mean + K.exp(log_std) * norm_sample


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    latent_dim = 3
    n_epochs = 10
    batch_size = 512

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.
    # x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    encoder, decoder, vae, loss = build_vae(input_shape=x_train.shape[1:], latent_size=latent_dim,
                                            nodes=(1000, 500, 120))
    if not os.path.isfile('test.h5'):
        vae.compile(optimizer='adam', loss=loss)
        vae.fit(x_train, x_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, x_test))
        vae.save_weights('test.h5')
    else:
        vae.load_weights('test.h5')

    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    for c in np.unique(y_test):
        i = np.where(y_test==c)
        ax.scatter(x_test_encoded[i, 0], x_test_encoded[i, 1], x_test_encoded[i, 2], label=c)
    ax.legend()

    # Display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # We will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi, 0]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()
