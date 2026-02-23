import tensorflow as tf
from tensorflow.keras import layers, models

# Define the MLP model as per the provided PyTorch structure
def build_mlp(input_dim=47, hidden_dim=512, num_classes=12, dropout=0.10):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(num_classes, activation=None)(x)
    probs = layers.Activation('softmax')(logits)
    model = models.Model(inputs=inputs, outputs=[logits, probs])
    return model

# Build and compile the model
mlp_model = build_mlp(input_dim=47, hidden_dim=512, num_classes=12, dropout=0.10)
mlp_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Print the model summary
mlp_model.summary()