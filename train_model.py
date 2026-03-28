import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

# Dummy dataset
X = np.random.rand(50, 5, 64, 64, 1)
y = np.random.rand(50, 64, 64, 1)

model = Sequential()

model.add(ConvLSTM2D(
    filters=32,
    kernel_size=(3,3),
    input_shape=(5,64,64,1),
    padding='same',
    return_sequences=True
))

model.add(BatchNormalization())

model.add(ConvLSTM2D(
    filters=16,
    kernel_size=(3,3),
    padding='same',
    return_sequences=False
))

# ✅ FIXED LAYER
model.add(Conv2D(
    filters=1,
    kernel_size=(3,3),
    activation='sigmoid',
    padding='same'
))

model.compile(loss='mse', optimizer='adam')

model.fit(X, y, epochs=3)

model.save("convlstm_model.h5")

print("Model trained and saved!")