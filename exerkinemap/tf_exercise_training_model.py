import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential()

# Input and Reshape
model.add(layers.InputLayer(input_shape=(28, 28, 1)))
model.add(layers.Reshape((28, 28, 1)))

# First Convolutional Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# First MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Second MaxPooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Note: Transpose operation normally wouldn't be here in a standard ConvNet architecture. For simplicity, we will assume reshaping or transposing is meant for flattening purpose.
model.add(layers.Permute((2, 1, 3)))

# Flatten the tensor before Dense layers
model.add(layers.Flatten())

# First Fully Connected Layer
model.add(layers.Dense(128, activation='relu'))

# Second Fully Connected Layer
model.add(layers.Dense(10, activation='softmax'))

# Output Layer
output = model.output
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
