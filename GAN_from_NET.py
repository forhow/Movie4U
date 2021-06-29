import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.optimizers import Adam


noise_data = np.random.normal(0, 1, (32, 100))
# generated_images = 0.5 * generator.predict(np.random.normal(0, 1, (32, 100))) + 0.5

def show_images(generated_images,cnt, n=4, m=8, figsize=(9, 5)):
    f, axes = plt.subplots(n, m, figsize=figsize)
    # plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
    for i in range(0, n):
        for j in range(0, m):
            ax = axes[i][j]
            ax.imshow(generated_images[i * m + j][:, :, 0], cmap=plt.cm.bone)
            ax.grid(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
    plt.tight_layout()
    plt.savefig('./NET_ref/gan_digit_img_{}.png'.format(cnt))
    # plt.show()
    plt.close()


# show_images(0.5 * generator.predict(np.random.normal(0, 1, (32, 100))) + 0.5)


## create generator
generator_ = Sequential([
    Dense(128 * 7 * 7, activation="relu", input_shape=(100,)),
    Reshape((7, 7, 128)),

    BatchNormalization(momentum=0.8),  # what is batch normalization??
    UpSampling2D(),  # what is upsampling??
    Conv2D(128, kernel_size=3, padding="same"),
    Activation("relu"),

    BatchNormalization(momentum=0.8),
    UpSampling2D(),
    Conv2D(64, kernel_size=3, padding="same"),
    Activation("relu"),

    BatchNormalization(momentum=0.8),
    Conv2D(1, kernel_size=3, padding="same"),
    Activation("tanh"),
])

noise_input = Input(shape=(100,), name="noise_input")
generator = Model(noise_input, generator_(noise_input), name="generator")

generator_.summary()  # summary가 매우 유용하군요.

optimizer = Adam(0.0002, 0.5)
generator.compile(loss='binary_crossentropy', optimizer=optimizer)



### create discriminator
discriminator_ = Sequential([
    Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same"),
    LeakyReLU(alpha=0.2),
    Dropout(0.25),

    Conv2D(64, kernel_size=3, strides=2, padding="same"),
    ZeroPadding2D(padding=((0, 1), (0, 1))),
    LeakyReLU(alpha=0.2),
    Dropout(0.25),
    BatchNormalization(momentum=0.8),

    Conv2D(128, kernel_size=3, strides=2, padding="same"),
    LeakyReLU(alpha=0.2),
    Dropout(0.25),
    BatchNormalization(momentum=0.8),

    Conv2D(256, kernel_size=3, strides=1, padding="same"),
    LeakyReLU(alpha=0.2),
    Dropout(0.25),
    Flatten(),
    Dense(1, activation='sigmoid'),
])
image_input = Input(shape=(28, 28, 1), name="image_input")

discriminator = Model(image_input, discriminator_(image_input), name="discriminator")
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator_.summary()



### Combined Model
noise_input2 = Input(shape=(100,), name="noise_input2")
"""
model과 sequential의 차이는?? 
가설1: 레이어를 쌓는 것이 sequential 이라면, sequential을 쌓는 것이 model인가???

1) 다음 모델의 경우는 랜덤으로 만든 이미지로부터 학습해서 새로운 이미지를 만들어내는 generator의 데이터를 
2) discriminator가 분류하는 형식으로 진행된다. 
"""
combined = Model(noise_input2, discriminator(generator(noise_input2)))
combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

## read image
(X_train, _), (_, _) = mnist.load_data()
# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

## training
"""
- 이 코드에서는 fit을 사용한 것이 아니라, train_on_batch를 사용했음. 
- train_on_batch와의 차이점?을 구글에 검색해보니, 큰 차이가 없다고 하긴 하는데
    - train_on_batch의 경우, 넘겨 받은 데이터에 대해서 gradient vector를 계산해서 적용하고 끝내는 것이고(1epoch)
    - fit의 경우는 epoch과 batch_size를 한번에 모두 넘겨준다는 것 정도가 차이가 된다. 
- GAN의 경우, discriminator의 학습시 마다 generator가 생성하는 데이터가 변화하게 된다. 
    - 즉 처음부터 모든 데이터가 존재하고 이를 한번에 학습시키는 fit과는 다르게, 한번씩 업데이트를 할때마다 모델이 변화하므로, 
    - train_on_batch를 사용하는 것이 매우 합당함.
"""
batch_size = 256
half_batch = batch_size // 2


def train(epochs, print_step=10):
    history = []
    for epoch in range(epochs):
        # discriminator 트레이닝 단계
        #######################################################################3
        # 데이터 절반은 실제 이미지, 절반은 generator가 생성한 가짜 이미지
        # discriminator가 실제 이미지와 가짜 이미지를 구별하도록 discriminator를 트레이닝
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(X_train[np.random.randint(0, X_train.shape[0], half_batch)],
                                                   np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generator.predict(np.random.normal(0, 1, (half_batch, 100))),
                                                   np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # generator 트레이닝 단계
        #######################################################################3
        # 전부 generator가 생성한 가짜 이미지를 사용.
        # discriminator가 구별하지 못하도록 generator를 트레이닝

        """
        generator를 트레이닝할 때는, 반드시 discriminator가 필요함. 
        generator가 만든 image를 평가해야 하고, 그래야 feedback이 생겨서 generator가 학습됨. 
        따라서, generator는 combined model을 통해 학습시키는데, 이때, discriminator도 함께 학습되면 안되기 때문에
        discriminator.trainable 을 False로 변경시켜 둔다. 
        """
        noise = np.random.normal(0, 1, (batch_size, 100))
        discriminator.trainable = False
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
        # 기록
        record = (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], 100 * g_loss[1])
        history.append(record)
        if epoch % print_step == 0:
            print("%5d [D loss: %.3f, acc.: %.2f%%] [G loss: %.3f, acc.: %.2f%%]" % record)
            show_images(0.5 * generator.predict(noise_data) + 0.5, cnt=epoch)
    return history


# %%time, 은
history100 = train(5000, 100)
# show_images(0.5 * generator.predict(noise_data) + 0.5)