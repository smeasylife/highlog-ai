import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("AWS_S3_BUCKET", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("JWT_SECRET", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")

from app.services.vector_service import VectorService


class FakePdfDoc:
    def __init__(self, total_pages):
        self.total_pages = total_pages

    def __len__(self):
        return self.total_pages

    def close(self):
        pass


class FakeDb:
    def __init__(self):
        self.bulk_data = []
        self.commits = 0
        self.rollbacks = 0

    def bulk_insert_mappings(self, model, bulk_data):
        self.bulk_data = bulk_data

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class VectorizePdfProgressTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self):
        service = VectorService.__new__(VectorService)

        async def embed_batch(texts):
            return [[float(i)] * 768 for i, _ in enumerate(texts)]

        service._embed_batch = embed_batch
        return service

    async def test_vectorize_pdf_streams_completed_batches_and_reorders_results(self):
        service = self.make_service()
        db = FakeDb()
        progress_events = []

        async def parse_batch(pdf_bytes, pages):
            import asyncio

            if pages == [0, 1]:
                await asyncio.sleep(0.03)
                return [{"index": 0, "text": "batch-1", "category": "세특"}]
            if pages == [2, 3]:
                await asyncio.sleep(0.01)
                return [{"index": 0, "text": "batch-2", "category": "창체"}]
            await asyncio.sleep(0.02)
            return [{"index": 0, "text": "batch-3", "category": "행특"}]

        async def progress_callback(progress):
            progress_events.append(progress)

        service._parse_pdf_batch_with_gemini = parse_batch

        with patch("fitz.open", return_value=FakePdfDoc(total_pages=5)):
            success, message, saved_count = await service.vectorize_pdf(
                pdf_bytes=io.BytesIO(b"%PDF test"),
                record_id=123,
                db=db,
                progress_callback=progress_callback,
            )

        self.assertTrue(success)
        self.assertEqual(saved_count, 3)
        self.assertEqual(message, "3 chunks successfully vectorized")
        self.assertEqual([row["chunk_text"] for row in db.bulk_data], ["batch-1", "batch-2", "batch-3"])
        self.assertEqual(progress_events, [10, 33, 56, 80, 80, 85, 95])

    async def test_vectorize_pdf_keeps_partial_success_when_one_batch_fails(self):
        service = self.make_service()
        db = FakeDb()
        progress_events = []

        async def parse_batch(pdf_bytes, pages):
            if pages == [0, 1]:
                raise RuntimeError("ocr failed")
            return [{"index": 0, "text": "batch-2", "category": "창체"}]

        async def progress_callback(progress):
            progress_events.append(progress)

        service._parse_pdf_batch_with_gemini = parse_batch

        with patch("fitz.open", return_value=FakePdfDoc(total_pages=4)):
            success, message, saved_count = await service.vectorize_pdf(
                pdf_bytes=io.BytesIO(b"%PDF test"),
                record_id=123,
                db=db,
                progress_callback=progress_callback,
            )

        self.assertTrue(success)
        self.assertEqual(saved_count, 1)
        self.assertIn("1 batches failed but skipped", message)
        self.assertEqual([row["chunk_text"] for row in db.bulk_data], ["batch-2"])
        self.assertEqual(progress_events, [10, 45, 80, 80, 85, 95])

    async def test_vectorize_pdf_fails_when_all_batches_fail_after_ocr_reaches_80(self):
        service = self.make_service()
        db = FakeDb()
        progress_events = []

        async def parse_batch(pdf_bytes, pages):
            raise RuntimeError("ocr failed")

        async def progress_callback(progress):
            progress_events.append(progress)

        service._parse_pdf_batch_with_gemini = parse_batch

        with patch("fitz.open", return_value=FakePdfDoc(total_pages=4)):
            success, message, saved_count = await service.vectorize_pdf(
                pdf_bytes=io.BytesIO(b"%PDF test"),
                record_id=123,
                db=db,
                progress_callback=progress_callback,
            )

        self.assertFalse(success)
        self.assertEqual(message, "Failed to generate chunks from all batches")
        self.assertEqual(saved_count, 0)
        self.assertEqual(db.bulk_data, [])
        self.assertEqual(db.commits, 0)
        self.assertEqual(progress_events, [10, 45, 80, 80])


if __name__ == "__main__":
    unittest.main()
