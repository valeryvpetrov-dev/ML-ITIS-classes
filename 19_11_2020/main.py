import os
import sys
import time
from datetime import datetime

import numpy as np
import pygame
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def configure_ssl():
    requests.packages.urllib3.disable_warnings()
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context


def mnist_cnn_model():
    image_size = 28
    num_channels = 1  # 1 for grayscale images
    num_classes = 10  # Number of outputs
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     padding='same',
                     input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def mnist_cnn_train(model):
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

    # Get image size
    image_size = 28
    num_channels = 1  # 1 for grayscale images

    # re-shape and re-scale the images data
    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
    train_data = train_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

    # re-shape and re-scale the images validation data
    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    # encode the labels - we have 10 output classes
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

    print("Training the network...")
    t_start = time.time()

    # Start training the network
    model.fit(train_data, train_labels_cat, epochs=8, batch_size=64,
              validation_data=(val_data, val_labels_cat))

    print("Done, dT:", time.time() - t_start)

    return model


def cnn_digits_predict(model, image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file,
                                             target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))

    result = model.predict_classes([img_arr])
    return result[0]


def predict_digit(model, surface):
    digit_image_path = './data/digit.png'
    pygame.image.save(surface, digit_image_path)
    digit_prediction = cnn_digits_predict(model, digit_image_path)
    draw_digit_prediction(surface, digit_prediction)
    os.remove(digit_image_path)


def draw_digit_prediction(surface, digit_prediction):
    font = pygame.font.SysFont(None, 30)
    img = font.render('Prediction: {}'.format(digit_prediction), True, BLUE)
    surface.blit(img, (5, 5))


def clear_screen(screen):
    screen.fill(WHITE)
    pygame.display.flip()


def save_current_picture(screen):
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pygame.image.save(screen, './data/{}.png'.format(now_str))


def start_predict_digit_game(cnn_predict_digit_model):
    pygame.init()

    drawing = False
    screen = pygame.display.set_mode((200, 200), 0, 32)
    screen.fill(WHITE)
    pygame.display.set_caption("ScratchBoard")

    last_pos = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mouse_position = pygame.mouse.get_pos()
                    if last_pos is not None:
                        pygame.draw.line(screen, BLACK, last_pos, mouse_position, 10)
                    last_pos = mouse_position
            elif event.type == pygame.MOUSEBUTTONUP:
                last_pos = None
                drawing = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    predict_digit(cnn_predict_digit_model, screen)
                elif event.key == pygame.K_c:
                    clear_screen(screen)
                elif event.key == pygame.K_s:
                    save_current_picture(screen)
        pygame.display.update()


if __name__ == '__main__':
    # create model
    # configure_ssl()
    # model = mnist_cnn_model()
    # mnist_cnn_train(model)
    # model.save('./model/cnn_digits_28x28.h5')

    # load trained model
    model = tf.keras.models.load_model('./model/cnn_digits_28x28.h5')
    start_predict_digit_game(model)
