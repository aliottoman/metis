"""The vendored appkit is Metis code and is tested as Metis code.

Each class of assertion here is a defect from a real generated build, now
pinned: float drift in money checks, zero-coerced missing values, trusted
client MIME labels, temp files leaking on refusal, background streams
mistaken for responses, and parse failures swallowed into empty objects.
"""

from __future__ import annotations

import asyncio
import io
from decimal import Decimal
from pathlib import Path

import pytest

from waqil_api.scaffold.appkit import config as appkit_config
from waqil_api.scaffold.appkit import money, uploads
from waqil_api.scaffold.appkit.oci_responses import (
    ExtractionError,
    OciResponses,
    parse_json_output,
)

PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
JPEG_HEAD = b"\xff\xd8\xff\xe0" + b"\x00" * 24


# --- config -----------------------------------------------------------------


def test_require_names_the_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCI_RESPONSES_PROJECT_ID", raising=False)
    with pytest.raises(appkit_config.ConfigError, match="OCI_RESPONSES_PROJECT_ID"):
        appkit_config.require("OCI_RESPONSES_PROJECT_ID")


def test_load_dotenv_fills_but_never_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nAPPKIT_TEST_A=from-file\nAPPKIT_TEST_B='quoted'\nbroken-line\n"
    )
    monkeypatch.setenv("APPKIT_TEST_A", "from-environment")
    monkeypatch.delenv("APPKIT_TEST_B", raising=False)
    appkit_config.load_dotenv(env_file)
    assert appkit_config.optional("APPKIT_TEST_A") == "from-environment"
    assert appkit_config.optional("APPKIT_TEST_B") == "quoted"


def test_from_env_reads_lazily_not_at_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The module imported fine at the top of this file with no OCI settings —
    # that is the import-time half. The use-time half fails clearly:
    monkeypatch.chdir(tmp_path)  # no .env to load
    for name in (
        "OCI_RESPONSES_BASE_URL",
        "OCI_RESPONSES_PROJECT_ID",
        "OCI_RESPONSES_MODEL_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(appkit_config.ConfigError, match="OCI_RESPONSES_BASE_URL"):
        appkit_config.OciResponsesConfig.from_env()


# --- money ------------------------------------------------------------------


def test_money_survives_float_drift() -> None:
    assert money.to_money(0.1 + 0.2) == Decimal("0.30")
    assert money.sum_money([0.1, 0.2]) == Decimal("0.30")


def test_missing_stays_missing_not_zero() -> None:
    assert money.to_money(None) is None
    assert money.sum_money([170, None]) is None
    assert money.within_cents(None, 100) is None
    assert money.within_percent(100, None, 5) is None


def test_one_cent_tolerance_is_inclusive() -> None:
    assert money.within_cents("500.00", "500.01") is True
    assert money.within_cents("500.00", "500.02") is False
    assert money.within_cents("500.00", "500.00") is True


def test_five_percent_boundary_inclusive_and_one_cent_past_fails() -> None:
    assert money.within_percent("100.00", "105.00", 5) is True
    assert money.within_percent("100.00", "105.01", 5) is False
    assert money.within_percent("100.00", "95.00", 5) is True
    assert money.within_percent("100.00", "105.00", 5, inclusive=False) is False


def test_unreadable_values_are_missing() -> None:
    assert money.to_money("not a number") is None
    assert money.to_money("1,234.50") == Decimal("1234.50")


# --- uploads ----------------------------------------------------------------


class _Upload:
    """The async-read shape of a FastAPI UploadFile, over fixed bytes."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


def _run(coro):  # noqa: ANN001, ANN202 - tiny test helper
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_sniff_trusts_bytes_not_labels() -> None:
    assert uploads.sniff_mime(PNG_HEAD) == "image/png"
    assert uploads.sniff_mime(JPEG_HEAD) == "image/jpeg"
    assert uploads.sniff_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert uploads.sniff_mime(b"plain text") == "application/octet-stream"


def test_save_upload_accepts_and_removes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    saved = _run(uploads.save_upload(_Upload(PNG_HEAD)))
    assert saved.mime == "image/png"
    assert saved.size == len(PNG_HEAD)
    assert saved.path.exists()
    assert saved.path.parent.resolve() == tmp_path.resolve()
    assert "upload-" in saved.path.name  # generated name, never the client's
    saved.remove()
    assert not saved.path.exists()
    saved.remove()  # idempotent


def test_save_upload_refuses_oversize_without_leaking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(uploads.UploadError, match="limit"):
        _run(uploads.save_upload(_Upload(PNG_HEAD * 100), max_bytes=64))
    assert list(tmp_path.iterdir()) == []  # nothing left behind


def test_save_upload_refuses_wrong_type_and_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(uploads.UploadError, match="unsupported"):
        _run(uploads.save_upload(_Upload(b"%PDF-1.7 not an image")))
    with pytest.raises(uploads.UploadError, match="empty"):
        _run(uploads.save_upload(_Upload(b"")))
    assert list(tmp_path.iterdir()) == []


# --- oci_responses ----------------------------------------------------------


class _Response:
    def __init__(self, status: str, *, output_text: str = "", error: object = None):
        self.status = status
        self.id = "resp-1"
        self.output_text = output_text
        self.error = error


class _FakeResponses:
    def __init__(self, response: _Response, *, delay: float = 0.0) -> None:
        self._response = response
        self._delay = delay
        self.create_kwargs: dict[str, object] = {}

    async def create(self, **kwargs: object) -> _Response:
        self.create_kwargs = kwargs
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._response


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


def _adapter(fake: _FakeResponses) -> OciResponses:
    adapter = OciResponses(
        appkit_config.OciResponsesConfig(
            base_url="https://example.invalid/openai/v1",
            project_id="ocid1.aiproject.oc1.test",
            model_id="xai.grok-4.3",
            profile="DEFAULT",
            config_file="",
        )
    )
    adapter._client = _FakeClient(fake)  # inject; _get_client returns it as-is
    return adapter


def test_extract_document_builds_the_verified_input_shape_synchronously() -> None:
    fake = _FakeResponses(_Response("completed", output_text='{"total": 525}'))
    text = _run(_adapter(fake).extract_document("extract it", PNG_HEAD))
    assert text == '{"total": 525}'
    assert fake.create_kwargs["model"] == "xai.grok-4.3"
    # Signed OCI requests cannot be re-executed by the service, so the call
    # is synchronous by contract: no background, no stream — verified live,
    # a backgrounded job dies on "invalid authentication header".
    assert "background" not in fake.create_kwargs
    assert "stream" not in fake.create_kwargs
    (message,) = fake.create_kwargs["input"]  # type: ignore[misc]
    assert message["role"] == "user"
    text_part, image_part = message["content"]
    assert text_part == {"type": "input_text", "text": "extract it"}
    assert image_part["type"] == "input_image"
    assert image_part["image_url"].startswith("data:image/png;base64,")


def test_run_preserves_terminal_failure_detail() -> None:
    class _Error:
        message = "capacity exceeded"

    fake = _FakeResponses(_Response("failed", error=_Error()))
    with pytest.raises(ExtractionError) as caught:
        _run(_adapter(fake).generate("hello"))
    assert caught.value.status == "failed"
    assert "capacity exceeded" in caught.value.detail


def test_run_times_out_as_a_typed_error() -> None:
    fake = _FakeResponses(_Response("completed", output_text="late"), delay=0.2)
    with pytest.raises(ExtractionError) as caught:
        _run(_adapter(fake).generate("hello", timeout_seconds=0.01))
    assert caught.value.status == "timeout"


def test_empty_completed_response_is_an_error_not_blank_data() -> None:
    fake = _FakeResponses(_Response("completed", output_text=""))
    with pytest.raises(ExtractionError, match="empty"):
        _run(_adapter(fake).generate("hello"))


def test_parse_json_output_handles_fences_and_prose_but_never_guesses() -> None:
    assert parse_json_output('{"a": 1}') == {"a": 1}
    assert parse_json_output('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_output('The result is {"a": 1} as requested.') == {"a": 1}
    with pytest.raises(ExtractionError, match="not valid JSON"):
        parse_json_output("I could not read the document, sorry.")


def test_parse_json_output_names_a_missing_await() -> None:
    """The failure a live build actually produced: the coroutine went straight
    into the parser, and the only symptom was "'coroutine' object has no
    attribute 'strip'" raised from inside the scaffold."""

    async def _reply() -> str:
        return "{}"

    coro = _reply()
    try:
        with pytest.raises(ExtractionError, match="await"):
            parse_json_output(coro)
    finally:
        coro.close()

    with pytest.raises(ExtractionError, match="got dict"):
        parse_json_output({"already": "parsed"})
