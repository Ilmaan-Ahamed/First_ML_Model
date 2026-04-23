# ETH-80 Image Classification

A clean and structured TensorFlow/Keras project for classifying images from the ETH-80 dataset.

## Project Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow & Keras to classify images from the ETH-80 dataset.

The project demonstrates a complete Deep Learning pipeline, including:

- Dataset download & extraction
- Data preprocessing & augmentation
- CNN model building
- Training & validation
- TensorBoard visualization
- Model saving & prediction

## Dataset

**Dataset**: ETH-80 Image Dataset

**Classes**: 8 categories (e.g., dog, horse, etc.)

**Structure**:
```
eth-80/
├── train_set/
└── val_set/
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- PIL (Image Processing)
- TensorBoard

## Setup Instructions

### 1️⃣ Install Dependencies
```powershell
pip install tensorflow numpy pillow matplotlib tensorboard
```

### 2️⃣ Download & Extract Dataset
```python
import urllib.request
import tarfile
import os

url = 'https://github.com/sadeepj/eth-80/releases/download/0.0.1/eth-80.tar.gz'
dest = 'eth-80.tar.gz'

if not os.path.exists(dest):
    urllib.request.urlretrieve(url, dest)

if not os.path.isdir('eth-80'):
    with tarfile.open(dest, 'r:gz') as tar:
        tar.extractall()
```

### 3️⃣ Verify Dataset
```python
from PIL import Image
from IPython.display import display

im = Image.open('eth-80/train_set/dog/dog4/dog4-066-207.png')
display(im)
```

## Model Configuration

- **Image Dimensions**: 128x128
- **Epochs**: 50
- **Batch Size**: 32

## CNN Model Architecture
```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(8, activation='softmax'))
```

## Data Preprocessing
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

## Data Generators
```python
train_generator = train_datagen.flow_from_directory(
    'eth-80/train_set',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'eth-80/val_set',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)
```

## Model Training
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=2952 // 32,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=328 // 32
)
```

## TensorBoard Visualization
```powershell
tensorboard --logdir logs/fit
```

## Save & Load Model
```python
model.save_weights('saved_weights.weights.h5')
model.load_weights('saved_weights.weights.h5')
```

## Prediction
```python
import numpy as np
from PIL import Image

im = Image.open('eth-80/val_set/horse/horse10-066-117.png')
img = np.array(im) / 255.
img = img[np.newaxis, ...]

prediction = model.predict(img)
print(np.argmax(prediction))
```

## Project Structure
```
Deep-Learning-CNN/
│
├── eth-80/                         # Dataset (ignored in GitHub)
│   ├── train_set/
│   └── val_set/
│
├── logs/                           # TensorBoard logs
│
├── CNNs_With_keras.ipynb           # Main notebook
├── eth-80.tar.gz                   # Dataset archive
├── saved_weights.weights.h5        # Model weights
│
├── LICENSE
└── README.md
```

## Output

- Model predicts one of 8 classes
- Output is a probability distribution
- `argmax()` returns predicted class index

## Future Improvements

- 🔁 Apply Transfer Learning (ResNet, VGG16)
- ⚙️ Hyperparameter tuning
- 🌐 Deploy using Streamlit / Flask
- 📊 Add accuracy & loss visualization

## Author

**Ilmaan Ahamed**  
Software Engineering Undergraduate  
SLTC Research University

## Support

If you like this project, give it a ⭐ on GitHub!
