from preprocess import *
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, CuDNNLSTM, Permute, Input, Reshape
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

inp = Input(shape=(config.buckets, config.max_len, channels))
reshape_out = Reshape((config.buckets, config.max_len))(inp)
perm_out = Permute((2,1))(reshape_out)
lstm_out = CuDNNLSTM(64)(perm_out)
dense_out = Dense(num_classes, activation="softmax")(lstm_out)
model = Model(inp, dense_out)

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
model.summary()
config.total_params = model.count_params()


model.fit(X_train, y_train_hot, batch_size=config.batch_size, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])

