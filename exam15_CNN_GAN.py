import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

'Setting'
OUT_DIR = './CNN_OUT_img/'
img_shape = (28,28,1)
epoch = 5000
batch_size = 128
noise = 100
sample_interval = 100

'Data load'
(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)


# scaling : range  in -1 ~ 1
X_train = X_train / 127.5 - 1

# reshape과 같은 역할, 차원 추가
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)


'build Generator'
generator_model = Sequential()
generator_model.add(Dense(128*7*7, input_shape=(noise,), activation='relu'))
generator_model.add(Reshape((7,7,128)))

# conv2dtranspose : autoencoder 작성시 사용한 upsampling + conv2d이 합쳐진 layer
# upsampling 과정에서 이미지가 2배로 증가 (14*14*128)
generator_model.add(BatchNormalization(momentum=0.8))
generator_model.add(Conv2DTranspose(128,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    activation='relu'))
# generator_model.add(LeakyReLU(alpha=0.01))

# BatchNormalization : 데이터 발산을 방지하기 위해 정규화
# loss의 최저점을 찾아야 하는데 지역저점에 고립되는 경우(값의 발산)을 방지하기 위해
# 같은 픽셀의 값들을 더한 값을 정규화 함
generator_model.add(BatchNormalization())

generator_model.add(Conv2DTranspose(64,
                                    kernel_size=3,
                                    strides=1,
                                    padding='same',
                                    activation='relu'))
# generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(BatchNormalization())

generator_model.add(Conv2DTranspose(1,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same'))
# -1~1사이 값으로 활성화 조건 부여
generator_model.add(Activation('tanh'))

generator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(generator_model.summary())


'build Discriminator'
discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size=3, strides=2,
                               padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.1))
discriminator_model.add(Dropout(0.25))

discriminator_model.add(Conv2D(64, kernel_size=3, strides=2,
                               padding='same'))
discriminator_model.add(ZeroPadding2D(padding=((0,1), (0,1))))
discriminator_model.add(LeakyReLU(alpha=0.1))
discriminator_model.add(Dropout(0.25))
# discriminator_model.add(BatchNormalization(momentum=0.8))

discriminator_model.add(Conv2D(128,kernel_size=3, strides=2,
                               padding='same'))
discriminator_model.add(LeakyReLU(alpha=0.1))
discriminator_model.add(Dropout(0.25))
# discriminator_model.add(BatchNormalization(momentum=0.8))

discriminator_model.add(Conv2D(256,kernel_size=3, strides=1,
                               padding='same'))
discriminator_model.add(LeakyReLU(alpha=0.1))
discriminator_model.add(Dropout(0.25))

discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))

discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(discriminator_model.summary())

# discriminator_model.trainable = False

'build GAN'
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
gan_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
print(gan_model.summary())

'Training Target'
real = np.ones((int(batch_size/2), 1))
fake = np.zeros((int(batch_size/2), 1))
print(real.shape)
print(fake.shape)

'test'
# for itr in range(30):
#     # idx = np.random.randint(0, X_train.shape[0], batch_size)
#     # print(type(idx))
#     # print(idx.shape)
#     # real_imgs = X_train[idx]
#     # print(real_imgs.shape)
#     f = np.random.normal(0, 1, (batch_size, noise))
#     fake_imgs = generator_model.predict(f)
#     print(fake_imgs.shape)
'test'

for itr in range(epoch):
    # 0~ X_train.shape[0] 사이의 값을 batch_size 개수 만큼 추출
    idx = np.random.randint(0, X_train.shape[0], int(batch_size/2))
    real_imgs = X_train[idx]

    # 평균 0, 표준편자 1인 배열 128*100개 생성
    z = np.random.normal(0, 1, (int(batch_size/2), noise))
    fake_imgs = generator_model.predict(z)

    # real img에 대한 정답(1), fake img에 대한 정답 (0) 학습
    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)

    # loss와 acc의 평균값 계산
    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    z_real = np.ones((batch_size, 1))
    # for i in range(2):
    #discriminator 학습 제한
    discriminator_model.trainable = False

    # noise data 생성
    z = np.random.normal(0,1,(batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, z_real)

    if itr % sample_interval ==0:
        print('%d [D loss : %f, acc.: %.2f] [G Loss : %f]'%(itr, d_loss, d_acc*100, gan_hist[0]))
        row = col = 4
        z = np.random.normal(0,1,(row*col, noise))
        fake_imgs = generator_model.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col), sharex= True, sharey=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt], cmap='gray')
                axs[i, j].axis('off') # x, y 축 모두 미출력
                cnt += 1
        path = os.path.join(OUT_DIR, 'img_{}'.format(itr))
        plt.savefig(path)
        plt.close()

