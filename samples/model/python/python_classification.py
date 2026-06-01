"""InferRT Python binding classification sample."""

from __future__ import annotations

import argparse
import time

import numpy as np

from util import (
    allocate_output_tensors,
    ensure_module_path,
    load_labels,
    preprocess_images,
    resolve_project_root,
    split_path_list,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run InferRT classification from Python bindings")
    parser.add_argument("-m", "--model", default="resnet18", help="Built-in model name")
    parser.add_argument("-w", "--weights", default="samples/model/classification/resnet18.wts", help="Path to .wts file")
    parser.add_argument("-i", "--image", default="assets/pics/dog.jpg", help="Input image path(s), comma-separated")
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
    image_paths = [project_root / path for path in split_path_list(args.image)]
    if not image_paths:
        raise ValueError("At least one input image is required")
    label_path = project_root / args.labels

    config = irt.ModelConfig()
    if len(image_paths) > 1:
        config.dynamic_batch_range = [1, len(image_paths), len(image_paths)]

    print(f"Creating model: {model_name}")
    model = irt.create_model(model_name, config)
    model.log_level = irt.LogLevel.INFO

    print(f"Building or loading engine from: {weights_path}")
    model.build_or_load(str(weights_path))

    input_tensor_names = model.input_tensor_names()
    output_tensor_names = model.output_tensor_names()
    print(f"Input tensors: {input_tensor_names}")
    print(f"Output tensors: {output_tensor_names}")

    if not input_tensor_names:
        raise RuntimeError("Model has no configured input tensors")
    if not output_tensor_names:
        raise RuntimeError("Model has no configured output tensors")

    if len(input_tensor_names) != 1:
        raise RuntimeError(
            f"This classification sample expects exactly one input tensor, got: {input_tensor_names}"
        )

    input_name = input_tensor_names[0]
    input_shape = list(model.tensor_shape(input_name))
    input_shape[0] = len(image_paths)
    model.set_tensor_shape(input_name, input_shape)

    preprocess_start = time.perf_counter()
    input_tensor = preprocess_images(image_paths, image_size=(int(input_shape[3]), int(input_shape[2])))
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    print(f"Running inference for batch size {len(image_paths)}")

    input_tensors = {input_name: input_tensor}
    output_tensors = allocate_output_tensors(model, output_tensor_names)

    infer_start = time.perf_counter()
    model.infer(input_tensors, output_tensors)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    output_name = output_tensor_names[0]
    output = output_tensors[output_name]
    print(f"Using output tensor for classification scores: {output_name}")

    postprocess_start = time.perf_counter()
    labels = load_labels(label_path) if label_path.exists() else []
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    print(
        f"Timing: preprocess={preprocess_ms:.3f} ms, "
        f"inference={infer_ms:.3f} ms, postprocess={postprocess_ms:.3f} ms"
    )
    batched_output = output.reshape((len(image_paths), -1)) if output.shape[0] == len(image_paths) else output.reshape(1, -1)
    for batch_index, scores in enumerate(batched_output):
        topk = min(args.topk, scores.shape[0])
        indices = np.argsort(scores)[::-1][:topk]
        print(f"\nTop results for batch {batch_index}:")
        for rank, index in enumerate(indices, start=1):
            score = float(scores[index])
            if labels and index < len(labels):
                print(f"top: {rank}, score: {score:.6f}, label[{index}]: {labels[index]}")
            else:
                print(f"top: {rank}, score: {score:.6f}, label[{index}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
