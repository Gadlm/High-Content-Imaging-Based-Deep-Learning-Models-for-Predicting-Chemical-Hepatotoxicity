def _tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ImportError("TensorFlow is required for CNN") from error
    return tf


def build_cnn(n_features, params, seed):
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    filters = list(params["conv_filters"])
    dropout = list(params["dropout"])
    inputs = tf.keras.Input((n_features, 1), name="phenotypic_profile")
    x = inputs
    for index, width in enumerate(filters):
        x = tf.keras.layers.Conv1D(width, params["kernel_size"], padding="same", activation="relu", name=f"conv_{index + 1}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index + 1}")(x)
        x = tf.keras.layers.MaxPooling1D(params["pool_size"], name=f"pool_{index + 1}")(x)
        x = tf.keras.layers.Dropout(dropout[index], name=f"dropout_{index + 1}")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(params["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout[-1])(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="probability")(x)
    model = tf.keras.Model(inputs, outputs, name="cnn")
    model.compile(tf.keras.optimizers.Adam(params["learning_rate"]), "categorical_crossentropy", metrics=["accuracy"])
    return model
