import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    Activation,
    Dropout,
    Add,
    MaxPooling1D,
    Dense,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam


def simclr_loss(_, hidden):
    temperature = 0.1
    large_num = 1e9
    hidden = tf.math.l2_normalize(hidden, -1)

    hidden1, hidden2 = tf.split(hidden, 2, axis=0)
    batch_size = tf.shape(hidden1)[0]

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    logits_aa -= masks * large_num

    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_bb -= masks * large_num

    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature

    loss_a = tf.keras.losses.categorical_crossentropy(
        labels, tf.concat([logits_ab, logits_aa], axis=1), from_logits=True
    )
    loss_b = tf.keras.losses.categorical_crossentropy(
        labels, tf.concat([logits_ba, logits_bb], axis=1), from_logits=True
    )

    return tf.reduce_mean(loss_a + loss_b)


def simclr_accuracy(_, hidden):
    large_num = 1e9
    hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, axis=0)
    batch_size = tf.shape(hidden1)[0]

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True)
    logits_aa -= masks * large_num

    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True)
    logits_bb -= masks * large_num

    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True)
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True)

    acc_a = tf.keras.metrics.categorical_accuracy(
        labels, tf.concat([logits_ab, logits_aa], axis=1)
    )
    acc_b = tf.keras.metrics.categorical_accuracy(
        labels, tf.concat([logits_ba, logits_bb], axis=1)
    )

    return tf.reduce_mean(acc_a + acc_b) / 2


class ResidualUnit:
    def __init__(
        self,
        n_samples_out,
        n_filters_out,
        kernel_initializer="he_normal",
        dropout=0.2,
        kernel_size=17,
        preactivation=True,
        postactivation_bn=False,
        activation="relu",
    ):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation

    def _skip_connection(self, y, downsample, n_filters_in):
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding="same")(y)
        if n_filters_in != self.n_filters_out:
            y = Conv1D(
                self.n_filters_out, 1, padding="same", use_bias=False,
                kernel_initializer=self.kernel_initializer,
            )(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]

        y = self._skip_connection(y, downsample, n_filters_in)

        x = Conv1D(
            self.n_filters_out, self.kernel_size, padding="same", use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        x = Conv1D(
            self.n_filters_out, self.kernel_size, strides=downsample,
            padding="same", use_bias=False, kernel_initializer=self.kernel_initializer,
        )(x)

        if self.preactivation:
            x = Add()([x, y])
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x

        return [x, y]


def get_model(projection_layer_shapes, dropout=0.2, ecg_len=4096, activation="relu", num_channels=12):
    kernel_size = 16
    kernel_initializer = "he_normal"
    signal = Input(shape=(ecg_len, num_channels), dtype=tf.float32, name="ecg")

    x = Conv1D(
        64, kernel_size, padding="same", use_bias=False,
        kernel_initializer=kernel_initializer,
    )(signal)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size, dropout=dropout, activation=activation)([x, x])
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size, dropout=dropout, activation=activation)([x, y])
    x, y = ResidualUnit(64, 256, kernel_size=kernel_size, dropout=dropout, activation=activation)([x, y])
    x, _ = ResidualUnit(16, projection_layer_shapes[0], kernel_size=kernel_size, dropout=dropout, activation=activation)([x, y])

    x = GlobalAveragePooling1D(name="embed")(x)

    for i, proj_size in enumerate(projection_layer_shapes[:-1]):
        x = Dense(proj_size, activation="relu", kernel_initializer=kernel_initializer, name=f"projection_{i}")(x)

    output = Dense(projection_layer_shapes[-1], kernel_initializer=kernel_initializer, name="projection")(x)

    return Model(inputs=signal, outputs=output)


def Pretrain_model(num_channels=12, num_samples=4096, num_feats=320) -> Model:
    model = get_model(
        projection_layer_shapes=[num_feats, num_feats],
        ecg_len=num_samples,
        dropout=0.0,
        activation="relu",
        num_channels=num_channels,
    )
    optimizer = Adam(1e-1)
    model.compile(
        loss=[simclr_loss],
        metrics=[simclr_accuracy],
        optimizer=optimizer,
    )
    return model


def get_model_embed(model_path, epoch=None, num_channels=12, num_samples=4096, num_feats=320) -> Model:
    fname = 'best_model' if epoch==0 else f'checkpoint_{epoch:04d}' 
    model_name = os.path.join(model_path, fname)

    model = Pretrain_model(num_channels, num_samples, num_feats)

    try:
        model.load_weights(model_name)
    except:
        model.load_weights(model_name + '.hdf5')

    # Use layer before final projection (remove SimCLR head)
    model_embed = Model(inputs=model.input, outputs=model.get_layer("embed").output)
    return model_embed
