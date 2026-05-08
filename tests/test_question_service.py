import asyncio
import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("AWS_S3_BUCKET", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("JWT_SECRET", "test")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")

from app.services.question_service import QuestionGenerationService


async def collect_updates(generator):
    updates = []
    async for update in generator:
        updates.append(update)
    return updates


class QuestionGenerationStreamingTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self):
        service = QuestionGenerationService.__new__(QuestionGenerationService)
        service.CATEGORIES = ["A", "B", "C"]
        return service

    async def test_guest_questions_stream_as_categories_complete_but_finish_in_category_order(self):
        service = self.make_service()
        delays = {"A": 0.03, "B": 0.01, "C": 0.02}

        async def process_single_category_from_chunks(**kwargs):
            category = kwargs["category"]
            await asyncio.sleep(delays[category])
            return {
                "success": True,
                "questions": [{"category": category, "content": f"question-{category}"}],
            }

        service._process_single_category_from_chunks = process_single_category_from_chunks

        updates = await collect_updates(
            service.generate_questions_from_chunks(
                record_chunks=[{"category": "A", "chunk_text": "text"}],
                target_school="school",
                target_major="major",
                interview_type="type",
            )
        )

        category_updates = updates[1:4]
        self.assertEqual([update["current_category"] for update in category_updates], ["B", "C", "A"])
        self.assertEqual([update["progress"] for update in category_updates], [33, 56, 79])
        self.assertEqual([update["completed_count"] for update in category_updates], [1, 2, 3])
        self.assertEqual(
            [question["content"] for question in updates[-1]["all_questions"]],
            ["question-A", "question-B", "question-C"],
        )

    async def test_guest_questions_stream_failure_immediately_and_report_ordered_failures(self):
        service = self.make_service()
        delays = {"A": 0.03, "B": 0.01, "C": 0.02}

        async def process_single_category_from_chunks(**kwargs):
            category = kwargs["category"]
            await asyncio.sleep(delays[category])
            if category == "B":
                raise RuntimeError("boom")
            return {
                "success": True,
                "questions": [{"category": category, "content": f"question-{category}"}],
            }

        service._process_single_category_from_chunks = process_single_category_from_chunks

        updates = await collect_updates(
            service.generate_questions_from_chunks(
                record_chunks=[{"category": "A", "chunk_text": "text"}],
                target_school="school",
                target_major="major",
                interview_type="type",
            )
        )

        first_category_update = updates[1]
        self.assertEqual(first_category_update["current_category"], "B")
        self.assertEqual(first_category_update["status_message"], "B 영역 실패 (1/3)")
        self.assertIn("1개 카테고리(B) 실패", updates[-1]["status_message"])
        self.assertEqual(
            [question["content"] for question in updates[-1]["all_questions"]],
            ["question-A", "question-C"],
        )

    async def test_record_questions_stream_as_categories_complete_but_finish_in_category_order(self):
        service = self.make_service()
        delays = {"A": 0.03, "B": 0.01, "C": 0.02}

        async def process_single_category(**kwargs):
            category = kwargs["category"]
            await asyncio.sleep(delays[category])
            return {
                "success": True,
                "questions": [{"category": category, "content": f"question-{category}"}],
            }

        service._process_single_category = process_single_category

        updates = await collect_updates(
            service.generate_questions(
                record_id=1,
                target_school="school",
                target_major="major",
                interview_type="type",
            )
        )

        category_updates = updates[1:4]
        self.assertEqual([update["current_category"] for update in category_updates], ["B", "C", "A"])
        self.assertEqual([update["progress"] for update in category_updates], [33, 56, 79])
        self.assertEqual(
            [question["content"] for question in updates[-1]["all_questions"]],
            ["question-A", "question-B", "question-C"],
        )


if __name__ == "__main__":
    unittest.main()
