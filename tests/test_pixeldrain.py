"""Transfer recovery exercises actual request bodies and streamed response failures."""

import hashlib
from unittest.mock import Mock

import pytest
import requests

from phantasm.pixeldrain import Pixeldrain


def response(status=200, data=None, chunks=(), headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = data
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.iter_content.return_value = iter(chunks)
    return result


def test_official_alternative_routes_authenticated_requests(monkeypatch):
    monkeypatch.setenv("PIXELDRAIN_DOMAIN", "pixeldrain.net")
    request = Mock(return_value=response(data={"files": []}))
    monkeypatch.setattr("requests.request", request)
    assert Pixeldrain("key").files() == []
    assert request.call_args.args == ("GET", "https://pixeldrain.net/api/user/files")
    assert request.call_args.kwargs["auth"] == ("", "key")
    assert request.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    "domain",
    [
        "",
        "example.com",
        "pixeldrain.net.evil.example",
        "http://pixeldrain.net",
        "pixeldrain.net/path",
    ],
)
def test_untrusted_domains_fail_before_sending_credentials(monkeypatch, domain):
    monkeypatch.setenv("PIXELDRAIN_DOMAIN", domain)
    request = Mock()
    monkeypatch.setattr("requests.request", request)
    with pytest.raises(ValueError, match="official"):
        Pixeldrain("key").files()
    request.assert_not_called()


def test_upload_rewinds_after_transient_http_failure(tmp_path, monkeypatch):
    source = tmp_path / "model.tar"
    source.write_bytes(b"artifact")
    checksum = hashlib.sha256(b"artifact").hexdigest()
    bodies = []

    def request(method, url, **kwargs):
        bodies.append(kwargs["data"].read())
        assert kwargs["auth"] == ("", "test-key")
        assert not kwargs["allow_redirects"]
        return (
            response(503)
            if len(bodies) == 1
            else response(201, {"success": True, "id": "abc", "size": 8, "hash_sha256": checksum})
        )

    monkeypatch.setattr("requests.request", request)
    monkeypatch.setattr("phantasm.pixeldrain.time.sleep", lambda _: None)
    receipt = Pixeldrain("test-key").upload(source)
    assert bodies == [b"artifact", b"artifact"]
    assert receipt["sha256"] == checksum


def test_upload_rejects_wrong_server_checksum(tmp_path, monkeypatch):
    source = tmp_path / "model.tar"
    source.write_bytes(b"artifact")
    monkeypatch.setattr(
        "requests.request",
        Mock(return_value=response(201, {"id": "abc", "size": 8, "hash_sha256": "0" * 64})),
    )
    with pytest.raises(RuntimeError, match="checksum"):
        Pixeldrain("key").upload(source)


def test_download_recovers_interrupted_stream_using_exact_range(tmp_path, monkeypatch):
    content = b"abcdefgh"
    checksum = hashlib.sha256(content).hexdigest()

    def interrupted():
        yield b"abc"
        raise requests.ConnectionError("private request detail")

    first = response(chunks=[])
    first.iter_content.return_value = interrupted()
    responses = iter(
        [
            response(data={"size": 8, "hash_sha256": checksum}),
            first,
            response(206, chunks=[b"defgh"], headers={"Content-Range": "bytes 3-7/8"}),
        ]
    )
    calls = []

    def request(method, url, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr("requests.request", request)
    monkeypatch.setattr("phantasm.pixeldrain.time.sleep", lambda _: None)
    out = Pixeldrain("key").download("abc", tmp_path / "result.tar")
    assert out.read_bytes() == content
    assert calls[-1]["headers"] == {"Range": "bytes=3-"}


def test_ignored_range_replaces_partial_bytes(tmp_path, monkeypatch):
    content = b"abcdef"
    checksum = hashlib.sha256(content).hexdigest()
    (tmp_path / f".result.tar.{checksum}.part").write_bytes(b"abc")
    monkeypatch.setattr(
        "requests.request",
        Mock(
            side_effect=[
                response(data={"size": 6, "hash_sha256": checksum}),
                response(chunks=[content]),
            ]
        ),
    )
    assert Pixeldrain("key").download("abc", tmp_path / "result.tar").read_bytes() == content


def test_invalid_range_never_commits_download(tmp_path, monkeypatch):
    checksum = hashlib.sha256(b"abcdef").hexdigest()
    monkeypatch.setattr(
        "requests.request",
        Mock(
            side_effect=[
                response(data={"size": 6, "hash_sha256": checksum}),
                response(206, chunks=[b"abcdef"], headers={"Content-Range": "bytes 1-5/6"}),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="range"):
        Pixeldrain("key").download("abc", tmp_path / "result.tar")
    assert not (tmp_path / "result.tar").exists()


def test_receipt_mismatch_and_existing_outputs_are_preserved(tmp_path, monkeypatch):
    info = response(data={"size": 6, "hash_sha256": "a" * 64})
    monkeypatch.setattr("requests.request", Mock(return_value=info))
    target = tmp_path / "result.tar"
    target.write_bytes(b"prior-result")
    with pytest.raises(RuntimeError, match="receipt"):
        Pixeldrain("key").download("abc", target, "b" * 64)
    with pytest.raises(ValueError, match="exists"):
        Pixeldrain("key").download("abc", target)
    assert target.read_bytes() == b"prior-result"


@pytest.mark.parametrize(
    "identifier", ["../secret", "https://other.example/file", "abc?token=secret"]
)
def test_arbitrary_download_urls_are_rejected(tmp_path, monkeypatch, identifier):
    request = Mock()
    monkeypatch.setattr("requests.request", request)
    with pytest.raises(ValueError):
        Pixeldrain("key").download(identifier, tmp_path / "out")
    request.assert_not_called()


def test_errors_do_not_expose_api_keys(monkeypatch):
    monkeypatch.setattr(
        "requests.request", Mock(side_effect=requests.RequestException("private-api-key"))
    )
    monkeypatch.setattr("phantasm.pixeldrain.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError) as exc:
        Pixeldrain("private-api-key").files()
    assert "private-api-key" not in str(exc.value)


def test_unverified_email_reports_required_action_without_retrying(tmp_path, monkeypatch):
    source = tmp_path / "artifact.tar"
    source.write_bytes(b"synthetic")
    request = Mock(
        return_value=response(
            403,
            {"value": "email_address_not_verified", "message": "untrusted-private-api-key"},
        )
    )
    monkeypatch.setattr("requests.request", request)
    with pytest.raises(RuntimeError, match="verify your Pixeldrain account email") as exc:
        Pixeldrain("private-api-key").upload(source)
    request.assert_called_once()
    assert "private-api-key" not in str(exc.value)


@pytest.mark.parametrize("error", [{"value": ["invalid"]}, {"value": "private-key"}, []])
def test_unknown_http_errors_do_not_echo_response_content(monkeypatch, error):
    monkeypatch.setattr("requests.request", Mock(return_value=response(403, error)))
    with pytest.raises(RuntimeError, match="check account access/limits") as exc:
        Pixeldrain("private-key").files()
    assert "private-key" not in str(exc.value)
