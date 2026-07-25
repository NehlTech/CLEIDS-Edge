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
# Baselines (CLEIDS_PROJECT_BRIEF.md §3), Notebook 04.
#
# Fidelity to source, stated explicitly per model (no fabricated hyperparameters --
# where a paper's real details couldn't be verified, that is stated, not hidden):
#   - build_standalone_cnn / build_standalone_lstm: EXACT ablations of
#     build_cleids_edge's own layer sizes above -- fully faithful by construction.
#   - build_altaie_hoomod2024: FULLY VERIFIED from the open-access paper itself
#     (Eng. Technol. Appl. Sci. Res. 14, 16740-16743, read directly from the PDF,
#     including its Table I hyperparameters and Figure 3 layer diagram).
#   - build_nazir2024_hybrid, build_misrak_melaku2025: paper is PAYWALLED
#     (Ain Shams Engineering Journal / Discover Internet of Things respectively) --
#     only abstract-level information could be verified (dataset, general
#     technique). Layer sizes below are REASONABLE DEFAULTS, not sourced from
#     the paper, stated explicitly here and again in Notebook 04's markdown.
#   - build_wang2023_dlbilstm: paper IS open access (PeerJ Computer Science) but
#     itself does not publish fixed hyperparameters -- it tunes them per-dataset
#     via Optuna rather than reporting fixed values. Layer sizes below are a
#     REASONABLE DEFAULT consistent with the paper's stated topology (DNN +
#     BiLSTM fusion, IPCA feature reduction), not a literal reproduction of an
#     Optuna-searched configuration that was never fixed in the source.
# ---------------------------------------------------------------------------

def build_random_forest(class_weight="balanced", n_estimators=200, random_state=42):
    """Baseline 1: Random Forest (classical ML floor).

    sklearn.ensemble.RandomForestClassifier. Not parameterized by input_dim --
    sklearn infers feature count from the training data at fit() time.
    class_weight="balanced" compensates for the same class imbalance SMOTE
    was applied for elsewhere in this project, without needing a second
    resampling pass on top of the already-SMOTE'd training split.
    """
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def build_svm(use_linear_fallback, class_weight="balanced", random_state=42):
    """Baseline 2: Support Vector Machine (classical ML floor).

    Kernel SVC (sklearn.svm.SVC) scales roughly O(n^2)-O(n^3) with training
    row count, which is impractical at this project's scale (post-SMOTE
    training sets range 504K-2.28M rows across the 5 datasets). Rather than
    silently downgrading, `use_linear_fallback` is decided explicitly by the
    caller (Notebook 04) per dataset based on training set size, and the
    substitution is documented there. When True, LinearSVC + CalibratedClassifierCV
    is used instead -- CalibratedClassifierCV is required because LinearSVC
    alone has no predict_proba, which the threshold-tuning step needs.
    """
    if use_linear_fallback:
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(class_weight=class_weight, random_state=random_state, max_iter=5000)
        return CalibratedClassifierCV(base, cv=3)
    else:
        from sklearn.svm import SVC
        return SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=random_state)


def build_standalone_cnn(input_dim):
    """Baseline 3: Standalone 1D-CNN (no LSTM component) -- CNN-branch
    ablation of build_cleids_edge, isolating its contribution to the hybrid's
    performance. Identical Conv1D block structure/filter sizes to
    build_cleids_edge (64 then 128 filters), with GlobalAveragePooling1D +
    Dense head replacing the LSTM branch entirely (not just removed --
    something has to reduce the (timesteps, 128) tensor to a vector, and
    global average pooling is the standard CNN-only substitute for that role).
    """
    inputs = keras.Input(shape=(input_dim, 1), name="input")

    x = layers.Conv1D(64, kernel_size=3, activation="relu", name="conv1d_block1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(128, kernel_size=3, activation="relu", name="conv1d_block2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = layers.Dropout(0.3, name="dropout_conv")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Standalone_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def build_standalone_lstm(input_dim):
    """Baseline 4: Standalone LSTM (no CNN component) -- LSTM-branch ablation
    of build_cleids_edge, isolating its contribution to the hybrid's
    performance. Same LSTM(100) size as build_cleids_edge's own LSTM layer,
    applied directly to the raw feature vector (reshaped as a length-input_dim
    sequence of 1 feature per step) rather than to CNN-extracted features.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="input")
    x = layers.LSTM(100, return_sequences=False, name="lstm")(inputs)
    x = layers.Dropout(0.3, name="dropout_lstm")(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Standalone_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def build_nazir2024_hybrid(input_dim):
    """Baseline 5: Nazir et al. (2024) hybrid CNN-LSTM architecture.

    Ain Shams Engineering Journal, 15, 102777 -- PAYWALLED (ScienceDirect);
    only abstract-level detail was verifiable (GWO feature selection, SMOTE,
    MinMax/standard normalization, ToN_IoT/UNSW-NB15 evaluation). Exact layer
    sizes, dropout, optimizer, and epoch count are NOT published in any
    accessible source -- the architecture below is a REASONABLE DEFAULT
    hybrid CNN-LSTM (2 conv blocks feeding a single LSTM, deeper than
    CLEIDS-Edge's own to reflect the paper's reported very high accuracy),
    not a verified reproduction. State this plainly in the Notebook 04
    markdown cell for this model, not just here.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="input")

    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = layers.LSTM(128, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Nazir2024_Hybrid")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def build_altaie_hoomod2024(input_dim):
    """Baseline 6: Altaie & Hoomod (2024) hybrid lightweight CNN+LSTM
    (Raspberry Pi-targeted).

    Eng. Technol. Appl. Sci. Res., 14, 16740-16743 -- FULLY OPEN ACCESS,
    read directly from the source PDF. Architecture below follows the
    paper's Figure 3 exactly: Conv1D(64, kernel=8) -> LSTM(256) ->
    Conv1D(64, kernel=8) -> LSTM(64) -> LSTM(32) -> Dense(256) -> Dense(64)
    -> softmax/sigmoid. Hyperparameters from the paper's Table I: epochs=30,
    batch_size=32, learning_rate=0.001, dropout=0.3 (train_and_evaluate call
    in Notebook 04 should use these, not this project's default 50/256).
    Not reproduced: the paper's PRESENT+SPECK feature-encryption step between
    its two phases -- that is a lightweight-cryptography detail orthogonal to
    classification accuracy, out of scope for a pure model-comparison benchmark.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="input")

    x = layers.Conv1D(64, kernel_size=8, padding="same", name="conv1")(inputs)
    x = layers.LeakyReLU(name="leaky_relu1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = layers.LSTM(256, return_sequences=True, name="lstm1")(x)

    x = layers.Conv1D(64, kernel_size=8, padding="same", name="conv2")(x)
    x = layers.LeakyReLU(name="leaky_relu2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = layers.LSTM(64, return_sequences=True, name="lstm2")(x)
    x = layers.LSTM(32, return_sequences=False, name="lstm3")(x)

    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AltaieHoomod2024")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def build_wang2023_dlbilstm(input_dim):
    """Baseline 7: Wang et al. (2023) "DL-BiLSTM" -- IPCA + dynamic
    quantization lightweight IDS.

    PeerJ Computer Science (open access), DOI 10.7717/peerj-cs.1569. The
    paper's own topology description ("DNN and BiLSTM fusion... dual hidden
    layer DNN") is followed here as: raw features treated as a pseudo-sequence
    fed to a BiLSTM (same input convention as the rest of this project) for
    temporal/bidirectional feature extraction, followed by a 2-hidden-layer
    DNN classifier head. NOT reproduced here: IPCA feature reduction (this
    project's shared preprocessing/feature set is fixed across all baselines
    per the evaluation protocol -- re-deriving per-model features would break
    the fair-comparison requirement) and the post-training 8-bit dynamic
    quantization (quantization is Notebook 05's scope, applied to CLEIDS-Edge
    specifically). Exact BiLSTM/DNN unit counts are NOT published in the
    paper (tuned per-dataset via Optuna instead of reported as fixed values)
    -- sizes below are a reasonable default, not a literal reproduction.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="input")
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="bilstm")(inputs)
    x = layers.Dense(128, activation="relu", name="dnn_hidden1")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="dnn_hidden2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Wang2023_DLBiLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def build_misrak_melaku2025(input_dim):
    """Baseline 8: Misrak & Melaku (2025) lightweight IDS with dynamic
    quantization.

    Discover Internet of Things, 5, 97 -- PAYWALLED (Springer); only
    abstract-level detail was verifiable (RAL-MIFS + two-stage IPCA feature
    selection, QAT/PTDQ quantization, a "DNN-BiLSTMQ" model evaluated on
    CIC-IDS2017/CIC-IoT2023). The "DNN-BiLSTMQ" name and abstract both
    indicate this extends Wang (2023)'s DNN-BiLSTM design (baseline 7) with
    improved feature engineering and quantization-aware training -- so the
    same topology as build_wang2023_dlbilstm is reused here, since no
    independently-verified architecture difference could be confirmed. The
    QAT/PTDQ quantization itself is out of scope for Notebook 04's
    standard-training comparison (quantization is Notebook 05's focus,
    applied to CLEIDS-Edge specifically) -- reasonable default, not verified.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="input")
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False), name="bilstm")(inputs)
    x = layers.Dense(128, activation="relu", name="dnn_hidden1")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="dnn_hidden2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MisrakMelaku2025")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model
