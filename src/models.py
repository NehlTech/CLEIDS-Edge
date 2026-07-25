"""CLEIDS-Edge model architectures.

build_cleids_edge() is parameterized by input_dim because the four (soon five)
benchmark datasets have different feature counts after per-dataset encoding
(NSL-KDD=122, CICIDS2017=78, UNSW-NB15=194, TON_IoT=76, IoT-23=pending) --
there is no single fixed input shape. Notebook 03 calls this once per dataset.

Baseline stubs (build_random_forest ... build_misrak_melaku2025) are interface
signatures only, for Notebook 04 to implement against -- see CLEIDS_PROJECT_BRIEF.md §3.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cleids_edge(input_dim, num_classes=2, binary=True):
    """Hybrid CNN-LSTM intrusion detection model.

    Deliberately lightweight (single small Dense head, no stacked LSTM) so it
    survives post-training INT8 quantization and magnitude pruning for edge
    deployment (Notebook 05).

    Args:
        input_dim: number of engineered features for the target dataset.
        num_classes: number of attack classes for the multiclass head
            (ignored when binary=True).
        binary: if True, single-sigmoid binary head (attack/benign); if
            False, softmax head over num_classes.

    Returns:
        A compiled tf.keras.Model.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="input")

    x = layers.Conv1D(64, kernel_size=3, activation="relu", name="conv1d_block1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(128, kernel_size=3, activation="relu", name="conv1d_block2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = layers.Dropout(0.3, name="dropout_conv")(x)

    x = layers.LSTM(100, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.3, name="dropout_lstm")(x)

    x = layers.Dense(64, activation="relu", name="dense_1")(x)

    if binary:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
        loss = "categorical_crossentropy"

    model = keras.Model(inputs=inputs, outputs=outputs, name="CLEIDS_Edge")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Baseline stubs (CLEIDS_PROJECT_BRIEF.md §3) -- signatures only.
# Full implementations happen in Notebook 04, re-implementing each as
# faithfully as published hyperparameters allow; missing details get a
# reasonable default noted there, not here.
# ---------------------------------------------------------------------------

def build_random_forest():
    """Baseline 1: Random Forest (classical ML floor).

    sklearn.ensemble.RandomForestClassifier. Not parameterized by input_dim --
    sklearn infers feature count from the training data at fit() time.
    Hyperparameters (n_estimators, max_depth, ...) chosen in Notebook 04.
    """
    raise NotImplementedError("Implemented in Notebook 04")


def build_svm():
    """Baseline 2: Support Vector Machine (classical ML floor).

    sklearn.svm.SVC. Not parameterized by input_dim, same reason as above.
    Kernel choice and hyperparameters chosen in Notebook 04.
    """
    raise NotImplementedError("Implemented in Notebook 04")


def build_standalone_cnn(input_dim):
    """Baseline 3: Standalone 1D-CNN (no LSTM component).

    Isolates the CNN branch's contribution to CLEIDS-Edge's hybrid
    performance. Same input_dim convention as build_cleids_edge.
    """
    raise NotImplementedError("Implemented in Notebook 04")


def build_standalone_lstm(input_dim):
    """Baseline 4: Standalone LSTM (no CNN component).

    Isolates the LSTM branch's contribution to CLEIDS-Edge's hybrid
    performance. Same input_dim convention as build_cleids_edge.
    """
    raise NotImplementedError("Implemented in Notebook 04")


def build_nazir2024_hybrid(input_dim):
    """Baseline 5: Nazir et al. (2024) hybrid CNN-LSTM architecture.

    Ain Shams Engineering Journal, 15, 102777. Re-implemented as faithfully
    as published hyperparameters allow in Notebook 04.
    """
    raise NotImplementedError("Implemented in Notebook 04")


def build_altaie_hoomod2024(input_dim):
    """Baseline 6: Altaie & Hoomod (2024) hybrid lightweight CNN+LSTM
    (Raspberry Pi-targeted).

    Eng. Technol. Appl. Sci. Res., 14, 16740-16743. Re-implemented in
    Notebook 04.
    """
    raise NotImplementedError("Implemented in Notebook 04")


def build_wang2023_dlbilstm(input_dim):
    """Baseline 7: Wang et al. (2023) "DL-BiLSTM" -- IPCA + dynamic
    quantization lightweight IDS.

    Re-implemented in Notebook 04.
    """
    raise NotImplementedError("Implemented in Notebook 04")


def build_misrak_melaku2025(input_dim):
    """Baseline 8: Misrak & Melaku (2025) lightweight IDS with dynamic
    quantization.

    Discover Internet of Things, 5, 97. Re-implemented in Notebook 04.
    """
    raise NotImplementedError("Implemented in Notebook 04")
