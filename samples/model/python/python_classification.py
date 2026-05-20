"""InferRT Python binding classification sample."""

from __future__ import annotations

import argparse

import numpy as np

from util import ensure_module_path, load_labels, preprocess_image, resolve_project_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run InferRT classification from Python bindings")
    parser.add_argument("-m", "--model", default="resnet18", help="Built-in model name")
    parser.add_argument("-w", "--weights", default="samples/model/classification/resnet18.wts", help="Path to .wts file")
    parser.add_argument("-i", "--image", default="assets/pics/dog.jpg", help="Input image path")
    parser.add_argument("-l", "--labels", default="assets/imagenet1000_clsidx_to_labels.txt", help="Label file path")
    parser.add_argument("-k", "--topk", type=int, default=3, help="Number of top results to print")
    parser.add_argument("-b", "--build-dir", default="build", help="CMake build directory used to locate .pyd and DLLs")
    return parser.parse_args()


def main() -> int:
    """Run the classification sample."""

    args = parse_args()
    project_root = resolve_project_root()
    build_dir = project_root / args.build_dir
    ensure_module_path(build_dir)

    import inferrt_model_py as irt

    model_name = args.model
    weights_path = project_root / args.weights
    image_path = project_root / args.image
    label_path = project_root / args.labels

    print(f"Creating model: {model_name}")
    model = irt.create_model(model_name)
    model.log_level = irt.LogLevel.INFO

    print(f"Building or loading engine from: {weights_path}")
    model.build_or_load(str(weights_path))

    input_tensor = preprocess_image(image_path)
    print(f"Running inference for image: {image_path}")
    output = model.infer(input_tensor)

    if not isinstance(output, np.ndarray):
        first_output_name = model.output_tensor_names()[0]
        output = output[first_output_name]

    scores = output.reshape(-1)
    topk = min(args.topk, scores.shape[0])
    indices = np.argsort(scores)[::-1][:topk]

    labels = load_labels(label_path) if label_path.exists() else []
    print("\nTop results:")
    for rank, index in enumerate(indices, start=1):
        score = float(scores[index])
        if labels and index < len(labels):
            print(f"top: {rank}, score: {score:.6f}, label[{index}]: {labels[index]}")
        else:
            print(f"top: {rank}, score: {score:.6f}, label[{index}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
