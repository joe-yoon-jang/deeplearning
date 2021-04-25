---
title: "GAN keras 예제코드"
last_modified_at: 2021-04-16T18:20:02-05:00
categories:
  - Study
tags:
  - Pythone
  - Deep Learning
  - GAN
---

# GAN Keras 샘플 코드

```python
import shutil
import os

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, Conv2D, \
    BatchNormalization, UpSampling2D, Reshape, Conv2DTranspose, ReLU
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

INPUT_SIZE = 100
PLOT_FRECUENCY = 50
```

```python
def read_image(file):
    image = open_image(file)
    image = normalize_image(image)
    return image


def open_image(file):
    image = Image.open(file)
    image = image.resize((64, 64))
    return np.array(image)


# Normalization, [-1,1] Range
def normalize_image(image):
    image = np.asarray(image, np.float32)
    image = image / 127.5 - 1
    return img_to_array(image)


# Restore, [0,255] Range
def denormalize_image(image):
    return ((image+1)*127.5).astype(np.uint8)
```
-  np.asarray: 이미 ndarray의 데이터 형태 (data type)이 설정 되어 있다면, 데이터 형태가 다를 경우에만 복사(copy) 가 된다.
- img_to_array: Keras API 로 Image 를 Numpy 배열로 변환
- Numpy 배열 ndarray의 메소드 astype(): 데이터형 dtype을 변경(캐스트)(uint8 부호없는8비트 정수)

```python
import matplotlib.pyplot as plt
def load_images():
    images = []        
    for cat in os.listdir('/kaggle/input/cat-and-dog/training_set/training_set/cats/'):
        try:
            image = read_image('/kaggle/input/cat-and-dog/training_set/training_set/cats/' + cat)
            images.append(image)
        except:
            print('No image', cat)

    return np.array(images)
x_train = load_images()
```

- 고양이 이미지 로드

```python
def create_generator():
    generator = Sequential()
    generator.add(Dense(units=256*4*4,input_dim=INPUT_SIZE))
    generator.add(Reshape((4,4,256)))

    generator.add(Conv2DTranspose(1024, 4, strides=1, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(512, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(256, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())

    generator.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(64, 4, strides=2, padding='same'))
    generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    generator.add(ReLU())
    
    generator.add(Conv2DTranspose(3, 3, strides=1, activation='tanh', padding='same'))
    
    generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))

    return generator

generator = create_generator()
generator.summary()
```

## Generator 생성
- Sequential() 모델생성
- add() 레이어를 쌓는다
- Dense
  - 보통의 밀집 연결 신경망 레이어
  - output = activation(dot(input, kernel) + bias) 
    - activation은 activation 인수로 전달되는 성분별 활성화 함수
    - kernel은 레이어가 만들어낸 가중치 행렬
    - bias는 레이어가 만들어낸 편향 벡터입니다 (use_bias가 True인 경우만 적용 가능합니다).
  - 인수
    - units: 양의 정수, 아웃풋 공간의 차원.
    - activation: 사용할 활성화 함수 (활성화를 참조하십시오). 따로 정하지 않으면, 활성화가 적용되지 않습니다. (다시 말해, "선형적" 활성화: a(x) = x).
    - use_bias: 불리언. 레이어가 편향 벡터를 사용하는지 여부.
    - kernel_constraint: kernel 가중치 행렬에 적용되는 제약 함수 (제약을 참조하십시오).
    - bias_constraint: 편향 벡터에 적용하는 제약 함수 (제약을 참조하십시오).  
  - 인풋 형태
    - (batch_size, ..., input_dim) 형태의 nD 텐서. 가장 흔한 경우는 (batch_size, input_dim) 형태의 2D 인풋입니다.
  - 아웃풋 형태
    - (batch_size, ..., units) 형태의 nD 텐서. 예를 들어, (batch_size, input_dim) 형태의 2D 인풋에 대해서 아웃풋은 (batch_size, units)의 형태를 갖게 됩니다.
- Reshape
  - keras.layers.Reshape(target_shape) 아웃풋을 특정 형태로 개조합니다.
  - 인수
    - target_shape: 표적 형태. 정수 튜플. 배치 축은 포함하지 않습니다.
  - 인풋 형태
    - 임의의 값. 하지만 개조된 인풋의 모든 차원은 고정되어야 합니다. 이 레이어를 모델의 첫 번째 레이어로 사용하는 경우, 키워드 인수 input_shape을 사용하십시오 (정수 튜플, 샘플 축을 포함하지 않습니다).
  - 아웃풋 형태
    - (batch_size,) + target_shape
- 배치 정규화(Batch Normalization)
  - 배치 정규화는 인공신경망에 입력값을 평균 0, 분산 1로 정규화(normalize)해 네트워크의 학습이 잘 일어나도록 돕는 방식
  - Inference Mode에서의 **BN은 O(1)**이므로, 상당히 Memory Bound되는 연산입니다.BN Fusing을 하게 되면, Conv이후 BN레이어를 수행하기 위해 RAM에 중간 Activation을 저장해야 할 필요가 없습니다. 따라서 램의 Bandwidth를 아껴 추론시 상당한 수행시간의 이득을 볼 수 있는 효과가 있습니다.
  - BN 레이어는 내부적으로 네가지 Parameter를 가집니다.
    - Mean
    - Variance
    - Gamma
    - Beta
- conv 
  - https://lsjsj92.tistory.com/416
  - https://zzsza.github.io/data/2018/02/23/introduction-convolution/

```python
def create_discriminator():
    discriminator = Sequential()

    discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding='same', input_shape=(64,64,3)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(1, kernel_size=4, strides=1, padding='same'))

    discriminator.add(Flatten())
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))
    return discriminator

discriminator = create_discriminator()
discriminator.summary()
```
```python
def create_gan(generator, discriminator):
    discriminator.trainable = False

    gan_input = Input(shape=(INPUT_SIZE,))
    generator_output = generator(gan_input)
    gan_output = discriminator(generator_output)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))

    return gan


gan = create_gan(generator, discriminator)
gan.summary()
```
```python
def plot_images(generator, size=25, dim=(5,5), figsize=(10,10)):
    noise= generate_noise(size)
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(denormalize_image(generated_images[i]), interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_loss(epoch, g_losses, d_losses):
    plt.figure(figsize=(10,5))
    plt.title("Loss, Epochs 0-" + str(epoch))
    plt.plot(g_losses,label="Generator")
    plt.plot(d_losses,label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
```
```python
def generate_noise(size):
    return np.random.normal(0, 1, size=[size, INPUT_SIZE])


def training(epochs=1, batch_size=32):
    #Loading Data
    batches = x_train.shape[0] / batch_size
    
    # Adversarial Labels
    y_valid = np.ones(batch_size)*0.9
    y_fake = np.zeros(batch_size)
    discriminator_loss, generator_loss = [], []

    for epoch in range(1, epochs+1):
        g_loss = 0; d_loss = 0
        print('epochs',epochs)
        for _ in range(int(batches)):
            # Random Noise and Images Set
            noise = generate_noise(batch_size)
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate Fake Images
            generated_images = generator.predict(noise)
            
            # Train Discriminator (Fake and Real)
            discriminator.trainable = True
            d_valid_loss = discriminator.train_on_batch(image_batch, y_valid)
            d_fake_loss = discriminator.train_on_batch(generated_images, y_fake)            

            d_loss += (d_fake_loss + d_valid_loss)/2
            
            # Train Generator
            noise = generate_noise(batch_size)
            discriminator.trainable = False
            g_loss += gan.train_on_batch(noise, y_valid)
            
        discriminator_loss.append(d_loss/batches)
        generator_loss.append(g_loss/batches)
            
        if epoch % PLOT_FRECUENCY == 0:
            print('Epoch', epoch)
            plot_images(generator)
            plot_loss(epoch, generator_loss, discriminator_loss)

    
training(epochs=100)
```
```python
def save_images(generator):
    if not os.path.exists('../output'):
        os.mkdir('../output')

    noise = generate_noise(10000)
    generated_images = generator.predict(noise)

    for i in range(generated_images.shape[0]):
        image = denormalize_image(generated_images[i])
        image = array_to_img(image)
        image.save( '../output/' + str(i) + '.png')

    shutil.make_archive('images', 'zip', '../output')
    
    
save_images(generator)
```