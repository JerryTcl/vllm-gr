# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.open_one_rec.one_rec_acc_test import run_api_inference


def test_completion_api_sends_rendered_prompt_verbatim(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"text": "first"}, {"text": "second"}]}

    def post(url, *, headers, json):
        calls.append((url, headers, json))
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    outputs = run_api_inference(
        ["<|im_start|>system\nrendered"],
        "model",
        "http://127.0.0.1:8000/v1/completions",
        "token",
        {
            "num_beams": 2,
            "num_return_sequences": 2,
            "max_new_tokens": 5,
            "temperature": 0.6,
            "top_p": 0.95,
        },
        max_concurrency=1,
    )

    assert calls[0][0] == "http://127.0.0.1:8000/v1/completions"
    assert calls[0][2]["prompt"] == "<|im_start|>system\nrendered"
    assert "messages" not in calls[0][2]
    assert [choice.text for choice in outputs[0].outputs] == ["first", "second"]


def test_chat_api_keeps_message_payload_and_response_parsing(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "reasoning-",
                            "content": "answer",
                        }
                    }
                ]
            }

    def post(url, *, headers, json):
        calls.append((url, headers, json))
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    outputs = run_api_inference(
        ["rendered"],
        "model",
        "http://127.0.0.1:8000/v1/chat/completions",
        "token",
        {
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 5,
            "temperature": 0.6,
            "top_p": 0.95,
        },
        max_concurrency=1,
    )

    assert calls[0][2]["messages"] == [{"role": "user", "content": "rendered"}]
    assert "prompt" not in calls[0][2]
    assert outputs[0].outputs[0].text == "reasoning-answer"
