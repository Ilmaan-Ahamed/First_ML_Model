import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_DIR = 'saved_model'
IMG_SIZE = (128, 128)


def load_model():
    model_path = os.path.join(MODEL_DIR, 'final_model')
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f'Model directory not found: {model_path}')
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path: str):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)


def main(image_path: str):
    model = load_model()
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    label_index = int(np.argmax(prediction, axis=1)[0])
    print('Prediction index:', label_index)
    print('Predicted score:', prediction[0, label_index])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load trained model and predict one image.')
    parser.add_argument('image_path', help='Path to an image file.')
    args = parser.parse_args()
    main(args.image_path)
