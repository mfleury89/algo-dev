import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from vae import build_vae, build_cnn_vae


def format_image(image, image_size=200):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.reshape(image, (image.shape[0], image.shape[1], 3))
    image = tf.image.resize(image, (image_size, image_size)) / 255.0
    # return tf.reshape(image, [-1])
    return image


if __name__ == '__main__':
    latent_dim = 50
    n_epochs = 100
    batch_size = 1
    image_size = 200

    image_files = glob.glob('images/*.jpg')
    n_images = len(image_files)
    # n_images = 10
    images = np.empty((n_images, image_size, image_size, 3))
    # images = np.empty((n_images, image_size * image_size * 3))
    for idx, image in enumerate(image_files[:n_images]):
        print("{}/{}".format(idx + 1, len(image_files)))
        im = cv2.imread(image)[:, :, [2, 1, 0]]
        images[idx, :] = format_image(im, image_size=image_size)

    tf.compat.v1.disable_eager_execution()

    encoder, decoder, vae, loss = build_cnn_vae(input_shape=images.shape[1:], latent_size=latent_dim)
    # encoder, decoder, vae, loss = build_vae(input_shape=images.shape[1:], latent_size=latent_dim,
    #                                         nodes=(1000,))
    if not os.path.isfile('test.h5'):
        vae.compile(optimizer='adam', loss=loss)
        vae.fit(images, images, epochs=n_epochs, batch_size=batch_size, validation_data=(images, images))
        vae.save_weights('test.h5')
    else:
        vae.load_weights('test.h5')

    x_test_encoded = encoder.predict(images, batch_size=batch_size)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_test_encoded[0][:, 0], x_test_encoded[0][:, 1], x_test_encoded[0][:, 2], label=0)
    ax.legend()

    plt.figure()
    x_test_decoded = decoder.predict(x_test_encoded[0])
    for x in x_test_decoded:
        image = (x.reshape(image_size, image_size, 3) * 255.0).astype(np.uint8)
        plt.imshow(image)
        plt.show()

    n = 10
    figure = np.zeros((image_size * n, image_size * n, 3), dtype=np.uint8)
    for idx in range(n):
        for jdx in range(n):
            if idx * n + jdx >= n_images:  # - 1:
                break
            # z_sample = np.expand_dims((x_test_encoded[0][idx * n + jdx, :] - x_test_encoded[0][idx * n + jdx + 1, :]) / 2, axis=0)
            z_sample = np.multiply(np.random.randn(1, latent_dim), x_test_encoded[1][idx * n + jdx, :]) + x_test_encoded[0][idx * n + jdx, :]
            x_decoded = decoder.predict(z_sample)
            image = (x_decoded[0].reshape(image_size, image_size, 3) * 255.0).astype(np.uint8)
            figure[idx * image_size: (idx + 1) * image_size,
            jdx * image_size: (jdx + 1) * image_size, :] = image

    plt.figure(figsize=(16, 16))
    plt.imshow(figure)
    plt.show()
