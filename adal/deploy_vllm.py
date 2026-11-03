from __future__ import annotations

import argparse
import shlex
import subprocess


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.vllm_executable, "serve", args.model,
        "--served-model-name", args.served_model_name,
        "--host", args.host, "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype, "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--seed", str(args.seed), "--generation-config", "vllm",
        "--reasoning-parser", "qwen3",
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Qwen3-14B with vLLM on the Linux GPU server.")
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen3-14B", help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--served-model-name", default="Qwen3-14B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--revision", default="", help="Optional pinned model revision")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    command = build_command(args)
    if args.print_command:
        print(shlex.join(command))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
