import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

''' 생성적 대립 신경망
Generative Adversarial Network,
- Generator
- Discriminator
'''

OUT_DIR = './OUT_img/'
img_shape = (28,28,1)
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100


(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)

# scaling : range  in -1 ~ 1
X_train = X_train / 127.5 - 1

# reshape과 같은 역할, 차원 추가
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)

# build generator
# LeakyRelu는 alpha parameter 전달필요해서 객체 생성 또는 레이어와 따로 지정
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise))
generator_model.add(LeakyReLU(alpha=0.01)) # activation function
# because use negative value
# LeakyReLu has small value on negative value range
'''
0<α<1 가  주어지면,  f는 다음과  같이 정의
$f(x) =max(αx, x) =  { x   if x ≥ 0}
                     { αx  if x < 0}
'''
generator_model.add(Dense(784, activation='tanh'))
generator_model.add(Reshape(img_shape)) # 28,28,1
print(generator_model.summary())

# build discriminator
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))
discriminator_model.add(Dense(128))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dense(1, activation='sigmoid'))
print(discriminator_model.summary())

discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

# disc' 모델에서 학습을 진행하지 않도록 설정
# disc 모델 compile시 학습속성과 gan에 로드되어 compile 될 때의 학습속성이 다름
discriminator_model.trainable = False

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())
gan_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

# Target으로 사용하기 위함
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))
print(real.shape)
print(fake.shape)

for itr in range(epoch):
    # 0~ X_train.shape[0] 사이의 값을 batch_size 개수 만큼 추출
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    # 평균 0, 표준편자 1인 배열 128*100개 생성
    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator_model.predict(z)

    # real img에 대한 정답(1), fake img에 대한 정답 (0) 학습
    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)

    # loss와 acc의 평균값 계산
    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)

    #discriminator 학습 제한
    discriminator_model.trainable = False

    # noise data 생성
    z = np.random.normal(0,1,(batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)

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
                axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off') # x, y 축 모두 미출력
                cnt += 1
        path = os.path.join(OUT_DIR, 'img_{}'.format(itr+1))
        plt.savefig(path)
        plt.close()

'''
model zoo.co
samsun neon
imma
rozy.gram
'''
