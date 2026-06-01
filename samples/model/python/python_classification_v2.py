"""InferRT DLPack-based classification sample using infer_v2."""

from __future__ import annotations

import argparse
import time

import numpy as np

from util import ensure_module_path, load_labels, preprocess_images, resolve_project_root, split_path_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InferRT classification with infer_v2 and DLPack tensors")
    parser.add_argument("-m", "--model", default="resnet18", help="Built-in model name")
    parser.add_argument("-w", "--weights", default="samples/model/classification/resnet18.wts", help="Path to .wts file")
    parser.add_argument("-i", "--image", default="assets/pics/dog.jpg", help="Input image path(s), comma-separated")
    parser.add_argument("-l", "--labels", default="assets/imagenet1000_clsidx_to_labels.txt", help="Label file path")
    parser.add_argument("-k", "--topk", type=int, default=3, help="Number of top results to print")
    parser.add_argument("-b", "--build-dir", default="build", help="CMake build directory used to locate .pyd and DLLs")
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="Tensor device for DLPack tensors")
    return parser.parse_args()


def torch_dtype_from_numpy_name(dtype_name: str) -> str:
    mapping = {
        "float32": "float32",
        "float16": "float16",
        "int8": "int8",
        "int32": "int32",
        "bool": "bool",
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported output dtype for torch allocation: {dtype_name}")
    return mapping[dtype_name]


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root()
    build_dir = project_root / args.build_dir
    ensure_module_path(build_dir)

    import torch
    import inferrt_model_py as irt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the infer_v2 DLPack sample")

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

    if len(input_tensor_names) != 1:
        raise RuntimeError(f"This sample expects exactly one input tensor, got: {input_tensor_names}")
    if not output_tensor_names:
        raise RuntimeError("Model has no configured output tensors")

    input_name = input_tensor_names[0]
    input_shape = list(model.tensor_shape(input_name))
    input_shape[0] = len(image_paths)
    model.set_tensor_shape(input_name, input_shape)

    preprocess_start = time.perf_counter()
    input_array = preprocess_images(image_paths, image_size=(int(input_shape[3]), int(input_shape[2])))
    input_tensor = torch.from_numpy(np.ascontiguousarray(input_array)).to(
        device=args.device, non_blocking=False
    ).contiguous()
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    output_tensors: dict[str, torch.Tensor] = {}
    for output_name in output_tensor_names:
        output_shape = model.tensor_shape(output_name)
        output_dtype = getattr(torch, torch_dtype_from_numpy_name(model.tensor_dtype(output_name)))
        output_tensors[output_name] = torch.empty(output_shape, dtype=output_dtype, device=args.device)

    stream = torch.cuda.Stream()
    input_tensors = {input_name: input_tensor}

    print(f"Running infer_v2 for batch size {len(image_paths)}")
    infer_start = time.perf_counter()
    with torch.cuda.stream(stream):
        model.infer_v2(input_tensors, output_tensors, stream_ptr=stream.cuda_stream, non_blocking=True)
    stream.synchronize()
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    postprocess_start = time.perf_counter()
    output_name = output_tensor_names[0]
    output = output_tensors[output_name].detach().cpu().numpy()
    print(f"Using output tensor for classification scores: {output_name}")

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
