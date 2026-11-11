# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the ordinary gRPC server with a local benchmark control service."""

import json

import grpc


def install_control_service():
    from vllm_gr.entrypoints.grpc import serve

    register = serve.rpc.add_VllmEngineServicer_to_server

    def add(servicer, server):
        register(servicer, server)

        async def control(request, context):
            command = request["command"]
            if command == "configure":
                return await servicer.engine.collective_rpc(
                    "configure_frontend_probe", args=(bool(request["enabled"]),)
                )
            if command == "drain":
                return await servicer.engine.collective_rpc("drain_frontend_probe")
            if command == "status":
                return await servicer.engine.collective_rpc("frontend_probe_status")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown benchmark command")

        handler = grpc.unary_unary_rpc_method_handler(
            control,
            request_deserializer=json.loads,
            response_serializer=lambda value: json.dumps(value).encode(),
        )
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "beam_frontend_benchmark.Control", {"Call": handler}
                ),
            )
        )

    serve.rpc.add_VllmEngineServicer_to_server = add


if __name__ == "__main__":
    from vllm_gr.plugin import initialize_runtime

    initialize_runtime()
    install_control_service()
    from vllm_gr.entrypoints.cli.main import main

    main()
