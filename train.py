import os
import tarfile
import urllib.request
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

DATA_URL = 'https://github.com/sadeepj/eth-80/releases/download/0.0.1/eth-80.tar.gz'
ARCHIVE_NAME = 'eth-80.tar.gz'
DATA_DIR = 'eth-80'
MODEL_DIR = 'saved_model'
LOG_DIR_BASE = 'logs/fit'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50


def download_and_extract_dataset():
    if os.path.isdir(DATA_DIR):
        print(f'Dataset already extracted: {DATA_DIR}')
        return

    if not os.path.exists(ARCHIVE_NAME):
        print(f'Downloading {ARCHIVE_NAME}...')
        urllib.request.urlretrieve(DATA_URL, ARCHIVE_NAME)
        print('Download complete.')

    print('Extracting dataset...')
    with tarfile.open(ARCHIVE_NAME, 'r:gz') as tar:
        tar.extractall()
    print('Extraction complete.')


def get_datasets():
    train_dir = os.path.join(DATA_DIR, 'train_set')
    val_dir = os.path.join(DATA_DIR, 'val_set')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=42,
    )

    # Store class names before pipeline operations
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    data_augmentation = models.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name='data_augmentation')

    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=IMG_SIZE + (3,)),
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ], name='eth80_classifier')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    download_and_extract_dataset()

    train_ds, val_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print('Class names:', class_names)

    model = build_model(num_classes)
    model.summary()

    log_dir = os.path.join(LOG_DIR_BASE, datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.h5'), save_best_only=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(os.path.join(MODEL_DIR, 'final_model'))
    print('Model saved to', os.path.join(MODEL_DIR, 'final_model'))

    return history


if __name__ == '__main__':
    main()
