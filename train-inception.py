from preprocess import *
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, BatchNormalization, Input, Concatenate, LeakyReLU
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

config.max_len = 11
config.buckets = 20


# Save data to array file first
#save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels = ["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 50
config.batch_size = 100

num_classes = 3

X_train = X_train.reshape(
    X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(
    X_test.shape[0], config.buckets, config.max_len, channels)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

#model = Sequential()
def inception_block(ip):
    conv1x1_out = Conv2D(1, (1,1), padding='same', activation="relu")(ip)
    conv3x3_out = Conv2D(1, (3,3), padding='same', activation="relu")(ip)
    conv5x5_out = Conv2D(1, (5,5), padding='same', activation="relu")(ip)
    maxpool3x3_out = MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same')(ip)
    concat_out = Concatenate()([conv1x1_out, conv3x3_out, conv5x5_out, maxpool3x3_out])
    return concat_out

inp = Input(shape=(config.buckets, config.max_len, channels))
#reshape_out = Reshape((img_width, img_height, 1))(inp)
incept1_out = inception_block(inp)
maxpool1_out = MaxPooling2D((2,2))(incept1_out)
incept2_out = inception_block(maxpool1_out)
batchnorm1_out = BatchNormalization()(incept2_out)
flat1_out = Flatten()(batchnorm1_out)
dense1_out = Dense(128, activation="relu")(flat1_out)
dropout1_out = Dropout(0.25)(dense1_out)
dense2_out = Dense(64, activation="relu")(dropout1_out)
dropout2_out = Dropout(0.2)(dense2_out)
dense3_out = Dense(num_classes, activation="softmax")(dropout2_out)
model = Model(inp, dense3_out)

model.summary()


#model.add(Flatten(input_shape=(config.buckets, config.max_len, channels)))
#model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
config.total_params = model.count_params()


model.fit(X_train, y_train_hot, batch_size=config.batch_size, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])

