"""Phase 8 smoke tests. Run: python -m unittest discover -s tests -p 'test_phase8_smoke.py' -v

Tests intentionally avoid loading PyTorch checkpoints or executing AI inference.
"""
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class Phase8SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.app = app
        cls.app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'success')

    def test_page_routes(self):
        for path in ('/', '/detection', '/multispectral', '/analytics', '/reports'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, f'{path}: {response.status_code}')

    def test_detection_missing_upload(self):
        response = self.client.post('/detection', data={})
        self.assertEqual(response.status_code, 400)

    def test_detection_rejects_wrong_extension(self):
        from io import BytesIO
        response = self.client.post('/detection', data={
            'detection_image': (BytesIO(b'not an image'), 'sample.txt')
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_report_missing_id(self):
        response = self.client.post('/reports/generate/2147483647')
        self.assertEqual(response.status_code, 404)

    def test_detection_route_uses_post(self):
        rules = {r.rule: r.methods for r in self.app.url_map.iter_rules()}
        self.assertIn('POST', rules['/detection'])
        self.assertIn('GET', rules['/detection'])

if __name__ == '__main__':
    unittest.main()
