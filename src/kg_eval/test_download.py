"""Offline regression tests for interrupted graph/label downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from .utils import download, sha256_file


class Response:
    def __init__(self, chunks=(), error=None, status=200):
        self.chunks = chunks
        self.error = error
        self.status = status
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        if self.status >= 400:
            response = requests.Response()
            response.status_code = self.status
            raise requests.exceptions.HTTPError(response=response)

    def iter_content(self, size):
        yield from self.chunks
        if self.error is not None:
            raise self.error


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'labels.tsv'
        self.url = 'https://example.test/labels.tsv'
        sleep_patch = patch('src.kg_eval.utils.time.sleep')
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def test_interrupted_stream_restarts_without_appending_partial_bytes(self):
        failed = Response([b'partial-old-body'], requests.exceptions.ChunkedEncodingError('Response ended prematurely'))
        successful = Response([b'Entry\tGene Names\n', b'P12345\tGENE1\n'])

        def get(*args, **kwargs):
            self.assertFalse(self.path.exists())
            self.assertFalse(self.path.with_suffix('.tsv.source.json').exists())
            return failed if not failed.closed else successful

        self.path.with_suffix('.tsv.part').write_bytes(b'stale interrupted file')
        with patch('src.kg_eval.utils.requests.get', side_effect=get) as request:
            self.assertEqual(download(self.url, self.path), self.path)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(failed.closed and successful.closed)
        self.assertEqual(self.path.read_bytes(), b'Entry\tGene Names\nP12345\tGENE1\n')
        metadata = json.loads(self.path.with_suffix('.tsv.source.json').read_text())
        self.assertEqual(metadata['sha256'], sha256_file(self.path))
        self.assertEqual(metadata['url'], self.url)
        self.assertFalse(self.path.with_suffix('.tsv.part').exists())
        self.sleep.assert_called_once_with(2)

    def test_exhausted_retries_never_publish_a_partial_cache(self):
        responses = [Response([b'partial'], requests.exceptions.ChunkedEncodingError()) for _ in range(3)]
        with patch('src.kg_eval.utils.requests.get', side_effect=responses) as request:
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                download(self.url, self.path, max_attempts=3)
        self.assertEqual(request.call_count, 3)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix('.tsv.source.json').exists())
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [2, 4])

    def test_connection_failure_before_body_is_retried(self):
        with patch('src.kg_eval.utils.requests.get', side_effect=[requests.exceptions.ConnectTimeout(), Response([b'ok'])]):
            download(self.url, self.path)
        self.assertEqual(self.path.read_bytes(), b'ok')

    def test_transient_http_error_is_retried(self):
        with patch('src.kg_eval.utils.requests.get', side_effect=[Response(status=503), Response([b'ok'])]):
            download(self.url, self.path)
        self.assertEqual(self.path.read_bytes(), b'ok')

    def test_permanent_http_error_is_not_retried(self):
        with patch('src.kg_eval.utils.requests.get', return_value=Response(status=404)) as request:
            with self.assertRaises(requests.exceptions.HTTPError):
                download(self.url, self.path)
        request.assert_called_once()
        self.sleep.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_completed_cache_is_reused_and_hash_is_still_checked(self):
        with patch('src.kg_eval.utils.requests.get', return_value=Response([b'cached'])):
            download(self.url, self.path)
        with patch('src.kg_eval.utils.requests.get') as request:
            download(self.url, self.path)
            self.path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'Cached source differs'):
                download(self.url, self.path)
            request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
