## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow & Keras** to classify images from the **ETH-80 dataset**.

The project demonstrates a complete **Deep Learning pipeline**, including:
- Dataset download & extraction  
- Data preprocessing & augmentation  
- CNN model building  
- Training & validation  
- TensorBoard visualization  
- Model saving & prediction  

---

## 📊 Dataset

- **Dataset:** ETH-80 Image Dataset  
- **Classes:** 8 categories (e.g., dog, horse, etc.)  
- **Structure:**
```
eth-80/
├── train_set/
└── val_set/
```

---

## 🛠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- PIL (Image Processing)  
- TensorBoard  

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install tensorflow numpy pillow matplotlib tensorboard
```

---

## 📥 Download & Extract Dataset

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

---

## 🖼️ Verify Dataset

```python
from PIL import Image
from IPython.display import display

im = Image.open('eth-80/train_set/dog/dog4/dog4-066-207.png')
display(im)
```

---

## ⚡ Model Configuration

```python
img_width, img_height = 128, 128
epochs = 50
batch_size = 32
```

---

## 🧠 CNN Model Architecture

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

---
