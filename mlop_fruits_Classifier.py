
import os
import json
import argparse

import tensorflow as tf
from tensorflow.keras import layers, models

import mlflow
import mlflow.tensorflow


# ✅ Your current dataset root (based on your "exists: True" output)
DATA_ROOT = os.path.join("fruits", "fruits")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "validation")
TEST_DIR  = os.path.join(DATA_ROOT, "test")


def build_cnn(img_size: tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Simple, solid CNN for multi-class classification."""
    return models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ])


def load_datasets(img_size: tuple[int, int], batch_size: int, seed: int):
    """Load train/val/test from directory structure."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Speed pipeline
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds   = val_ds.cache().prefetch(autotune)
    test_ds  = test_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names, num_classes


def save_metadata(artifact_dir: str, class_names: list[str], img_size: tuple[int, int]):
    """Save class names + image size so inference is reproducible."""
    os.makedirs(artifact_dir, exist_ok=True)
    meta_path = os.path.join(artifact_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"class_names": class_names, "img_size": list(img_size)},
            f,
            indent=2
        )
    return meta_path


def main():
    parser = argparse.ArgumentParser()

    # Training config
    parser.add_argument("--img", default=100, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=123, type=int)

    # MLflow config
    parser.add_argument("--tracking_uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="fruit_classifier")
    parser.add_argument("--model_name", default="Fruit_CNN")

    # Artifacts (metadata file)
    parser.add_argument("--artifact_dir", default="artifacts")

    args = parser.parse_args()
    img_size = (args.img, args.img)

    # ✅ sanity checks
    print("Working dir:", os.getcwd())
    print("Train exists:", os.path.exists(TRAIN_DIR), TRAIN_DIR)
    print("Val exists:", os.path.exists(VAL_DIR), VAL_DIR)
    print("Test exists:", os.path.exists(TEST_DIR), TEST_DIR)

    if not (os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR) and os.path.exists(TEST_DIR)):
        raise FileNotFoundError(
            "Dataset folders not found. Check DATA_ROOT/TRAIN_DIR/VAL_DIR/TEST_DIR paths."
        )

    # Load data
    train_ds, val_ds, test_ds, class_names, num_classes = load_datasets(
        img_size=img_size, batch_size=args.batch, seed=args.seed
    )

    # Build model
    model = build_cnn(img_size, num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Save metadata
    meta_path = save_metadata(args.artifact_dir, class_names, img_size)

    # MLflow setup (Registry works when MLflow server is running)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # Autolog params/metrics (not models) - we register manually
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(run_name="cnn_train_register") as run:
        # log params
        mlflow.log_param("img_size", str(img_size))
        mlflow.log_param("batch_size", args.batch)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("classes", ",".join(class_names))

        # Train
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

        # Evaluate
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_acc))

        # log metadata artifact
        mlflow.log_artifact(meta_path, artifact_path="metadata")

        # ✅ Log + Register model (ONLY ONCE)
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name=args.model_name
        )

        print("\n✅ DONE")
        print("Run ID:", run.info.run_id)
        print("Registered model:", args.model_name)
        print("Test accuracy:", test_acc)
        print("\nOpen MLflow UI: http://127.0.0.1:5000 → Models →", args.model_name)


if __name__ == "__main__":
    main()

