"""Isolated-subprocess peak-memory worker for Notebook 06.

Run as: python measure_peak_memory_worker.py '<json spec>'
Prints a single line on success: PEAK_MB=<float>

Each invocation is a fresh Python process, so resource.getrusage().ru_maxrss
at the end IS this process's true peak RSS -- a real, per-model measurement.
This replaces Notebook 06's original in-process delta measurement, which
turned out to be broken: ru_maxrss is a cumulative, process-wide high-water
mark that never resets, so across 55 sequential benchmarks sharing one
process, only the very first model ever registered a nonzero delta (see
CLEIDS_PROJECT_BRIEF.md, Notebook 06 section, for the full real finding).

spec["kind"] selects what to load/run:
  "cleids_original"  -- spec: dataset, task
  "cleids_quantized" -- spec: dataset, task (rebuilds the 16x8 TFLite artifact)
  "keras_baseline"   -- spec: dataset, model_name
  "rf"               -- spec: dataset (retrains fresh, same as Notebook 06 main loop)
  "svm"              -- spec: dataset, use_linear_fallback (retrains fresh)
"""
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import resource
import numpy as np


def sample_indices(n_rows, n=40, seed=0):
    n = min(n, n_rows)
    return np.random.RandomState(seed).choice(n_rows, size=n, replace=False)


def main():
    spec = json.loads(sys.argv[1])
    repo_dir = spec["repo_dir"]
    os.chdir(repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, "src"))

    kind = spec["kind"]
    dataset = spec.get("dataset")

    if kind in ("cleids_original", "cleids_quantized", "keras_baseline"):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from tensorflow.keras import layers as klayers

    if kind == "cleids_original":
        task = spec["task"]
        ckpt_path = f"models/cleids_edge_{dataset}_{task}.keras"
        test_data = np.load(f"data/processed/{dataset}/test.npz")
        X = test_data["X_cnn"].astype(np.float32)
        model = tf.keras.models.load_model(ckpt_path)
        for ix in sample_indices(X.shape[0]):
            model.predict(X[ix:ix + 1], batch_size=1, verbose=0)

    elif kind == "cleids_quantized":
        task = spec["task"]
        binary = task == "binary"
        ckpt_path = f"models/cleids_edge_{dataset}_{task}.keras"
        test_data = np.load(f"data/processed/{dataset}/test.npz")
        val_data = np.load(f"data/processed/{dataset}/val.npz")
        X = test_data["X_cnn"].astype(np.float32)
        X_val = val_data["X_cnn"].astype(np.float32)
        input_dim = X.shape[1]
        with open(f"data/processed/{dataset}/label_classes.json") as f:
            num_classes = len(json.load(f)["classes"])

        def build_export_model(input_dim, num_classes, binary, batch_size):
            inputs = tf.keras.Input(batch_shape=(batch_size, input_dim, 1), name="input")
            x = klayers.Conv1D(64, kernel_size=3, activation="relu", name="conv1d_block1")(inputs)
            x = klayers.BatchNormalization(name="bn1")(x)
            x = klayers.MaxPooling1D(pool_size=2, name="pool1")(x)
            x = klayers.Conv1D(128, kernel_size=3, activation="relu", name="conv1d_block2")(x)
            x = klayers.BatchNormalization(name="bn2")(x)
            x = klayers.MaxPooling1D(pool_size=2, name="pool2")(x)
            x = klayers.Dropout(0.3, name="dropout_conv")(x)
            x = klayers.LSTM(100, return_sequences=False, unroll=True, name="lstm")(x)
            x = klayers.Dropout(0.3, name="dropout_lstm")(x)
            x = klayers.Dense(64, activation="relu", name="dense_1")(x)
            if binary:
                outputs = klayers.Dense(1, activation="sigmoid", name="output")(x)
            else:
                outputs = klayers.Dense(num_classes, activation="softmax", name="output")(x)
            return tf.keras.Model(inputs=inputs, outputs=outputs, name="CLEIDS_Edge_export")

        def make_representative_dataset(X_val, batch_size=1, n_calib=1024):
            n_calib = min(n_calib, X_val.shape[0])
            calib = X_val[:n_calib]
            n_padded = int(np.ceil(n_calib / batch_size)) * batch_size
            calib_padded = np.zeros((n_padded,) + X_val.shape[1:], dtype=np.float32)
            calib_padded[:n_calib] = calib

            def representative_dataset():
                for start in range(0, n_padded, batch_size):
                    yield [calib_padded[start:start + batch_size]]
            return representative_dataset

        model = tf.keras.models.load_model(ckpt_path)
        export_model = build_export_model(input_dim, num_classes, binary, 1)
        export_model.set_weights(model.get_weights())
        converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(X_val, 1)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_bytes = converter.convert()
        tflite_path = f"/tmp/_peakmem_{dataset}_{task}.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        for ix in sample_indices(X.shape[0]):
            sample = X[ix:ix + 1].astype(np.float32)
            interpreter.set_tensor(input_details[0]["index"], sample)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]["index"])
        os.remove(tflite_path)

    elif kind == "keras_baseline":
        model_name = spec["model_name"]
        ckpt_path = f"models/{model_name}_{dataset}_binary.keras"
        test_data = np.load(f"data/processed/{dataset}/test.npz")
        X = test_data["X_cnn"].astype(np.float32)
        model = tf.keras.models.load_model(ckpt_path)
        for ix in sample_indices(X.shape[0]):
            model.predict(X[ix:ix + 1], batch_size=1, verbose=0)

    elif kind == "rf":
        from models import build_random_forest
        train_data = np.load(f"data/processed/{dataset}/train.npz")
        test_data = np.load(f"data/processed/{dataset}/test.npz")
        X_train, y_train = train_data["X_flat"], train_data["y_bin"].astype(int)
        X = test_data["X_flat"].astype(np.float32)
        rf = build_random_forest()
        rf.fit(X_train, y_train)
        rf.set_params(n_jobs=1)
        for ix in sample_indices(X.shape[0]):
            rf.predict_proba(X[ix:ix + 1])

    elif kind == "svm":
        from models import build_svm
        train_data = np.load(f"data/processed/{dataset}/train.npz")
        test_data = np.load(f"data/processed/{dataset}/test.npz")
        X_train, y_train = train_data["X_flat"], train_data["y_bin"].astype(int)
        X = test_data["X_flat"].astype(np.float32)
        svm = build_svm(use_linear_fallback=spec["use_linear_fallback"])
        svm.fit(X_train, y_train)
        for ix in sample_indices(X.shape[0]):
            svm.predict_proba(X[ix:ix + 1])

    else:
        raise ValueError(f"unknown kind: {kind}")

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"PEAK_MB={peak_kb / 1024.0:.3f}")


if __name__ == "__main__":
    main()
