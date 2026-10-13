import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_agent import SYSTEM_PROMPT, get_prospect, send_prospect_email
from run import EXAMPLES


class SendProspectEmailDisqualificationTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={})
        self.from_rep = {"email": "rep@example.com", "name": "Test Rep"}

    def send(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Follow up",
            body="Let's connect.",
            runtime=self.runtime,
            from_rep=self.from_rep,
        )

    def test_blocks_disqualified_source_record(self):
        prospect = {"prospect_id": "LEAD-1", "email": "lead@example.com"}
        with patch.object(data_service, "get_prospect_record", return_value={"disqualified": True}), \
                patch("gtm_agent.gtm_agent.uuid.uuid4") as make_message_id:
            result = self.send(prospect)

        make_message_id.assert_not_called()
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": "lead@example.com",
        })

    def test_blocks_disqualified_passed_record_when_source_is_missing(self):
        prospect = {"prospect_id": "LEAD-2", "email": "lead@example.com", "disqualified": True}
        with patch.object(data_service, "get_prospect_record", return_value=None), \
                patch("gtm_agent.gtm_agent.uuid.uuid4") as make_message_id:
            result = self.send(prospect)

        make_message_id.assert_not_called()
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
            "to": "lead@example.com",
        })

    def test_sends_eligible_source_record(self):
        prospect = {"prospect_id": "LEAD-3", "email": "lead@example.com", "name": "Lead"}
        with patch.object(data_service, "get_prospect_record", return_value={"disqualified": False}):
            result = self.send(prospect)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "lead@example.com")

    def test_prompt_explains_disqualification_blocks(self):
        self.assertIn("Never email a prospect whose record has disqualified=true", SYSTEM_PROMPT)
        self.assertIn("tell the rep the prospect is disqualified and no email was sent", SYSTEM_PROMPT)
        self.assertNotIn("do not second-guess", SYSTEM_PROMPT)

    def test_run_examples_for_disqualified_leads_are_blocked(self):
        prompts = {
            lead_id: (user_id, prompt)
            for user_id, prompt in EXAMPLES
            for lead_id in ("LEAD-50001", "LEAD-50002", "LEAD-50003", "LEAD-50004", "LEAD-50005")
            if lead_id in prompt
        }
        self.assertEqual(len(prompts), 5)

        for lead_id, (user_id, prompt) in prompts.items():
            with self.subTest(lead_id=lead_id):
                prospect = get_prospect.func(lead_id)["prospect"]
                with patch("gtm_agent.gtm_agent.uuid.uuid4") as make_message_id:
                    result = self.send(prospect)
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["error"], "Prospect is disqualified; email not sent.")
                self.assertEqual(result["to"], prospect["email"])
                self.assertIn(lead_id, prompt)
                self.assertTrue(user_id.startswith("rep_"))
                make_message_id.assert_not_called()
                self.assertIn("disqualified", SYSTEM_PROMPT)
                self.assertIn("no email was sent", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
