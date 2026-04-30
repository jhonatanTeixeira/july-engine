import sys
import os
import unittest
import hashlib
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project to path
sys.path.append(os.path.abspath("."))

from july_engine.services.storage.cloud_path import CloudPath

class TestCloudPath(unittest.TestCase):
    def setUp(self):
        # Ensure test cleanup
        if os.path.exists("test_data"):
            import shutil
            shutil.rmtree("test_data")
        os.makedirs("test_data", exist_ok=True)

    def test_local_path(self):
        cp = CloudPath("test_data/local.txt")
        path = cp.__fspath__()
        self.assertTrue(os.path.isabs(path))
        self.assertIn("local.txt", path)

    def test_file_protocol(self):
        cp = CloudPath("file://test_data/local_proto.txt")
        path = cp.__fspath__()
        self.assertIn("local_proto.txt", path)

    @patch('fsspec.core.url_to_fs')
    def test_download_if_local_missing(self, mock_url_to_fs):
        mock_fs = MagicMock()
        mock_url_to_fs.return_value = (mock_fs, None)
        
        mock_fs.exists.return_value = True
        
        cp = CloudPath("s3://bucket/test_data/remote.txt")
        # Local path should include the bucket name if we just strip s3://
        local_path = os.path.abspath("bucket/test_data/remote.txt")
        if os.path.exists(local_path): os.remove(local_path)
        
        path = cp.__fspath__()
        
        mock_fs.get.assert_called_with("s3://bucket/test_data/remote.txt", local_path)
        self.assertEqual(path, local_path)

    @patch('fsspec.core.url_to_fs')
    def test_upload_if_cloud_missing(self, mock_url_to_fs):
        mock_fs = MagicMock()
        mock_url_to_fs.return_value = (mock_fs, None)
        
        mock_fs.exists.return_value = False # Cloud missing
        
        # We must create the file at the path CloudPath expects: bucket/test_data/new_local.txt
        local_path = os.path.abspath("bucket/test_data/new_local.txt")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f: f.write("content")
        
        cp = CloudPath("s3://bucket/test_data/new_local.txt")
        cp.__fspath__()
        
        mock_fs.put.assert_called_with(local_path, "s3://bucket/test_data/new_local.txt")

    @patch('fsspec.core.url_to_fs')
    @patch('os.path.getmtime')
    def test_sync_if_checksum_diff(self, mock_mtime, mock_url_to_fs):
        mock_fs = MagicMock()
        mock_url_to_fs.return_value = (mock_fs, None)
        
        # Local file at the expected bucket path
        local_path = os.path.abspath("bucket/test_data/sync.txt")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f: f.write("local content")
        
        # Cloud setup
        mock_fs.exists.return_value = True
        mock_fs.info.return_value = {
            "ETag": "different_checksum",
            "mtime": datetime(2026, 1, 1).timestamp()
        }
        
        # Mock mtimes
        mock_mtime.return_value = datetime(2025, 1, 1).timestamp() # Local is older
        
        cp = CloudPath("s3://bucket/test_data/sync.txt")
        cp.__fspath__()
        
        mock_fs.get.assert_called_with("s3://bucket/test_data/sync.txt", local_path)

if __name__ == '__main__':
    unittest.main()
