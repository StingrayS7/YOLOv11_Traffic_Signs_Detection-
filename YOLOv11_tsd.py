"""
Используя датасет Traffic Signs Detection дообучим модель YOLO v11 распознавать дорожные знаки в дорожном траффике. Так же протестируем дообученную модель на видео которое находится  внутри датасета.

---

# Загрузка библиотек, датасета и подготовка к обучению
"""

!pip install ultralytics

!wget https://storage.yandexcloud.net/academy.ai/CV/traffic_signs_detection.zip

!unzip -qo 'traffic_signs_detection.zip'

import ultralytics

from ultralytics import YOLO
import os
from PIL import Image
import cv2
from IPython.display import Video
import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


ultralytics.checks()

yaml_path = './car/data.yaml'

# Загрузка модели YOLO v11
model = YOLO('yolo11s.pt')

"""
Редактируем шапку файла data.yaml чтобы избежать ошибок с не обнаруженными изображениями"""

!pip install ruamel.yaml

import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2) #для корректного форматирования вывода


with open(yaml_path, 'r') as f:
    data = yaml.load(f)


# Сохраняем исходные пути
train_path = data.get('train')
val_path = data.get('val')
test_path = data.get('test')

# Устанавливаем новый путь
data['path'] = '../car'

# Изменяем относительные пути, если они были заданы в формате ../path
if train_path and train_path.startswith("../"):
    data['train'] = train_path.replace("../", "")
if val_path and val_path.startswith("../"):
    data['val'] = val_path.replace("../", "")
if test_path and test_path.startswith("../"):
    data['test'] = test_path.replace("../", "")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

"""Было:
```
train: ../train/images
val: ../valid/images
test: ../test/images
```

Стало:


```
train: train/images
val: valid/images
test: test/images

path: ../car
```

# Обучение

Эпох 30, пакет 32, размер изображений 416
"""

results = model.train(
      data=yaml_path,
      epochs=30,
      batch=32,
      imgsz=416, # размер изображений
      plots=True
)

"""# Оценка"""

def result_train(path):
  img = cv2.imread(path)
  plt.figure(figsize=(10,10), dpi=200)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # возвращаем изображение в rgb
  plt.imshow(img_rgb)

data_dir = './runs/detect/train'
img_dir = os.path.join(data_dir, 'train_batch*.jpg')
files = glob.glob(img_dir)
imgs = []
for image in files:
    result_train(image)

# матрица ошибок
img_dir = os.path.join(data_dir, 'confusion_matrix_normalized.png')
result_train(img_dir)

# прочие метрики
metric_dir = os.path.join(data_dir,'*curve*.png')
files_metric = glob.glob(metric_dir)
imgs_metric = []
for image in files_metric:
    result_train(image)

# Commented out IPython magic to ensure Python compatibility.
# tensor board
# %load_ext tensorboard
# %tensorboard --logdir='./runs/detect/train'

# загрузим картинку для предсказания
!wget 'https://drive.usercontent.google.com/download?id=1h2UuJlOlMIi4ww5rEOKf0V3j4OOVvx_K&export=download&authuser=1&confirm=t' -O 'pred.jpg'

# предсказание картинки
image_test = 'pred.jpg'

pred = model.predict(source=image_test,
                     imgsz=640)

test_image = pred[0].plot(line_width=2)
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # возвращаем изображение в rgb
plt.imshow(test_image_rgb)

# детекция по видео
pred_video = 'video.mp4'

video_output = model.predict(source=pred_video, conf=0.6, save=True)

from moviepy.editor import *

predicted_video = './runs/detect/train2/video.avi'
clip = VideoFileClip(predicted_video)
clip.ipython_display(width=960)