# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the extended SMG bindings with package-relative imports/private pools."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "vllm_gr/entrypoints/grpc"
    output = root / "generated"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{root / 'proto'}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            str(root / "proto/common.proto"),
            str(root / "proto/vllm_engine.proto"),
        ],
        check=True,
    )
    # common.proto defines messages only; protoc emits no service bindings here.
    (output / "common_pb2_grpc.py").unlink(missing_ok=True)
    for path in output.glob("*_pb2*.py"):
        text = path.read_text()
        for module in ("common_pb2", "vllm_engine_pb2"):
            text = text.replace(f"import {module} as ", f"from . import {module} as ")
        if path.name.endswith("_pb2.py"):
            # Preserve the original package/service names without colliding with
            # the unmodified SMG descriptor in applications importing both APIs.
            pool = "_descriptor_pool.DescriptorPool()"
            if path.name == "vllm_engine_pb2.py":
                text = text.replace(
                    "DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(",
                    "_pool = " + pool + "\n"
                    "_pool.AddSerializedFile(common__pb2.DESCRIPTOR.serialized_pb)\n"
                    "DESCRIPTOR = _pool.AddSerializedFile(",
                )
            else:
                text = text.replace("_descriptor_pool.Default()", pool)
            module = "vllm_gr.entrypoints.grpc.generated." + path.stem
            text = text.replace(f"'{path.stem}'", repr(module))
        path.write_text(text)


if __name__ == "__main__":
    main()
