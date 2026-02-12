import argparse
import json
import numpy as np
import tensorflow as tf
import mlflow


def load_metadata(meta_path="artifacts/metadata.json"):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["class_names"], tuple(meta["img_size"])


def load_model_from_registry(model_name, version=None,
                             tracking_uri="http://127.0.0.1:5000"):

    mlflow.set_tracking_uri(tracking_uri)

    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    model = mlflow.tensorflow.load_model(model_uri)
    return model, model_uri


def preprocess_image(img_path, img_size):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_name", default="Fruit_CNN")
    parser.add_argument("--version", default="1")
    parser.add_argument("--tracking_uri", default="http://127.0.0.1:5000")
    parser.add_argument("--metadata", default="artifacts/metadata.json")

    args = parser.parse_args()

    class_names, img_size = load_metadata(args.metadata)

    model, model_uri = load_model_from_registry(
        args.model_name,
        version=args.version,
        tracking_uri=args.tracking_uri
    )

    x = preprocess_image(args.image, img_size)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    print("Model URI:", model_uri)
    print("Prediction:", class_names[pred_idx])
    print("Confidence:", float(probs[pred_idx]))


if __name__ == "__main__":
    main()
