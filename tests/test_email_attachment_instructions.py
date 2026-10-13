import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class EmailAttachmentInstructionTests(unittest.TestCase):
    def test_pricing_deck_request_is_disclosed_when_unavailable(self):
        self.assertIn("no available tool can attach or link", SYSTEM_PROMPT)
        self.assertIn("in the email or your reply to the rep", SYSTEM_PROMPT)
        self.assertIn("attached, enclosed, included, or delivered", SYSTEM_PROMPT)
        self.assertIn("tell the rep they must send it separately", SYSTEM_PROMPT)
        self.assertIn(
            "Only claim requested material was included if a tool confirms",
            SYSTEM_PROMPT,
        )

    def test_ordinary_email_still_sends_without_attachment_claims(self):
        self.assertIn("always carry the request through and send the email", SYSTEM_PROMPT)
        self.assertIn("If the rep requests material", SYSTEM_PROMPT)
        body = "Could we schedule a demo next week?"
        result = send_prospect_email.func(
            prospect={"name": "Alex", "email": "alex@example.com"},
            subject="Demo availability",
            body=body,
            runtime=object(),
            from_rep={"name": "Taylor", "email": "taylor@example.com"},
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)


if __name__ == "__main__":
    unittest.main()
