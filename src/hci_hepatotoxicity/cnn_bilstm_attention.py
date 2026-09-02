def _tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ImportError("TensorFlow is required for CNN-BiLSTM") from error
    return tf


def build_cnn_bilstm_attention(n_features, params, seed):
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    filters = list(params["conv_filters"])
    units = list(params["bilstm_units"])
    dropout = list(params["dropout"])
    inputs = tf.keras.Input((n_features, 1), name="phenotypic_profile")
    x = inputs
    for index, width in enumerate(filters):
        x = tf.keras.layers.Conv1D(width, params["kernel_size"], padding="same", activation="relu", name=f"conv_{index + 1}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index + 1}")(x)
        x = tf.keras.layers.MaxPooling1D(params["pool_size"], name=f"pool_{index + 1}")(x)
        x = tf.keras.layers.Dropout(dropout[index], name=f"conv_dropout_{index + 1}")(x)
    for index, width in enumerate(units):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(width, return_sequences=True), name=f"bilstm_{index + 1}")(x)
        if index < len(units) - 1:
            x = tf.keras.layers.Dropout(dropout[index + len(filters)], name=f"bilstm_dropout_{index + 1}")(x)
    scores = tf.keras.layers.Dense(1, activation="tanh", name="attention_score")(x)
    weights = tf.keras.layers.Softmax(axis=1, name="attention_weights")(scores)
    context = tf.keras.layers.Dot(axes=(1, 1), name="attention_context")([weights, x])
    x = tf.keras.layers.Flatten()(context)
    x = tf.keras.layers.Dense(params["dense_units"], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(params["l2"]))(x)
    x = tf.keras.layers.Dropout(dropout[-1], name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="probability")(x)
    model = tf.keras.Model(inputs, outputs, name="cnn_bilstm_attention")
    model.compile(tf.keras.optimizers.Adam(params["learning_rate"]), "categorical_crossentropy", metrics=["accuracy"])
    return model
