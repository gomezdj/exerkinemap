import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential()

# Input layer, expects input of shape (28, 28, 1)
model.add(layers.InputLayer(input_shape=(28, 28, 1)))

# Reshape layer
model.add(layers.Reshape((29, 28, 28, 1)))

# First Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# First MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Second MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Transpose layer
model.add(layers.Permute((3, 2, 1)))

# Reshape layer
model.add(layers.Reshape((-1, 128)))

# First Fully Connected Layer with Matrix Multiplication and Bias Addition
model.add(layers.Dense(128, activation=None))
model.add(tf.keras.layers.Activation('relu'))

# Second Fully Connected Layer with Matrix Multiplication and Bias Addition
model.add(layers.Dense(10, activation=None))
model.add(tf.keras.layers.Activation('softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
