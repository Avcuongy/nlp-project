import argparse
from pathlib import Path
from joblib import load
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils.common import set_seed
from utils.config_loader import load_config


def create_fnn_model(input_dim: int, num_classes: int, config: dict):
    """Create Feedforward Neural Network model based on config parameters.

    Args:
        input_dim: Input dimension (vocabulary size)
        num_classes: Number of output classes
        config: Configuration dictionary containing model parameters

    Returns:
        Compiled Keras model
    """
    params = config.get("parameters", {})

    n_layers = params.get("n_layers", 3)
    l2_reg = params.get("l2_reg", 0.01)
    lr = params.get("lr", 0.001)

    # Build model
    model = keras.Sequential(name="FNN")

    # Input layer
    model.add(layers.Input(shape=(input_dim,), name="input"))

    # Hidden layers
    for i in range(1, n_layers + 1):
        hidden_units = params.get(f"hidden{i}", 64)
        dropout_rate = params.get(f"dropout{i}", 0.3)

        model.add(
            layers.Dense(
                hidden_units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f"hidden{i}",
            )
        )
        model.add(layers.Dropout(dropout_rate, name=f"dropout{i}"))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Deep Learning model")
    parser.add_argument(
        "--model",
        type=str,
        default="fnn",
        choices=["fnn"],
        help="Model type to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config file (optional)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save trained model (optional)",
    )

    args = parser.parse_args()

    # Ensure reproducibility
    set_seed(42, deterministic=True)

    # Setup paths
    root = Path(__file__).resolve().parents[1]

    # Read processed data
    train_path = root / "data" / "processed" / "train.csv"
    val_path = root / "data" / "processed" / "val.csv"
    vectorizer_path = root / "models" / "vectorizer.pkl"

    # Load config
    if args.config is None:
        config_path = root / "config" / "dl" / f"{args.model}.yaml"
    else:
        config_path = Path(args.config)

    config = load_config(str(config_path))
    params = config.get("parameters", {})

    print("=" * 80)
    print(f"TRAINING {args.model.upper()} MODEL".center(80))
    print("=" * 80)
    print(f"\nConfig: {config_path}")

    # Load processed data
    print("\n1. LOADING PROCESSED DATA...")
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_val = pd.read_csv(val_path, encoding="utf-8")
    print(f"\tTrain size: {len(df_train)}")
    print(f"\tValidation size: {len(df_val)}")

    # Load label encoder
    print("\n2. LOADING LABEL ENCODER...")
    label_encoder_path = root / "models" / "label_encoder.pkl"
    le = load(label_encoder_path)
    print(f"\tLoaded from {label_encoder_path}")
    print(f"\tClasses: {list(le.classes_)}")
    num_classes = len(le.classes_)

    # Transform labels
    y_train = le.transform(df_train["label"].astype(str))
    y_val = le.transform(df_val["label"].astype(str))

    # Load vectorizer
    print("\n3. LOADING VECTORIZER...")
    print(f"\tLoading from {vectorizer_path}...")
    vec = load(vectorizer_path)
    vocab_size = len(vec.get_feature_names_out())
    print(f"\tVocabulary size: {vocab_size}")

    # Vectorize text
    print("\n4. VECTORIZING...")
    X_train_vec = vec.transform(df_train["comment"])
    X_val_vec = vec.transform(df_val["comment"])

    # Convert sparse matrix to dense for neural network
    X_train_dense = X_train_vec.toarray()
    X_val_dense = X_val_vec.toarray()

    print(f"\tTrain shape: {X_train_dense.shape}")
    print(f"\tValidation shape: {X_val_dense.shape}")

    # Create model
    print(f"\n5. CREATING {args.model.upper()} MODEL...")
    if args.model == "fnn":
        model = create_fnn_model(vocab_size, num_classes, config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print("\n\tModel Architecture:")
    model.summary()

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7
        ),
    ]

    # Train model
    print(f"\n6. TRAINING {args.model.upper()} MODEL...")
    batch_size = params.get("batch_size", 32)
    epochs = params.get("epochs", 20)

    history = model.fit(
        X_train_dense,
        y_train,
        validation_data=(X_val_dense, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    if args.save_path:
        save_path = args.save_path
    else:
        model_dir = root / "models" / "dl"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(model_dir / f"{args.model}.keras")

    print(f"\n7. SAVING MODEL TO {save_path}...")
    model.save(save_path)
    print("\tModel saved successfully!")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
