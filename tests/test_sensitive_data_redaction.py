import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import gtm_agent


class SensitiveDataRedactionTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "name": "Alex Smith",
            "email": "alex@example.com",
            "annual_revenue": 50000000,
            "billing_qualification": {
                "tax_id": "123-45-6789",
                "date_of_birth": "1980-01-01",
                "card_on_file": "4111111111111111",
            },
        }

    def test_get_prospect_omits_sensitive_fields(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value=self.record):
            result = gtm_agent.get_prospect.func("LEAD-12345")

        prospect = result["prospect"]
        self.assertEqual(prospect["name"], "Alex Smith")
        self.assertEqual(prospect["email"], "alex@example.com")
        self.assertEqual(prospect["annual_revenue"], 50000000)
        self.assertNotIn("billing_qualification", prospect)

    def test_new_profile_omits_sensitive_fields(self):
        with (
            patch.object(gtm_agent.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(gtm_agent.data_service, "get_prospect_record", return_value=self.record),
            patch.object(gtm_agent.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(gtm_agent.data_service, "fetch_account_details", return_value=[]),
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=["AWS"]),
            patch.object(gtm_agent.data_service, "save_profile_to_db"),
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-12345")

        profile = result["prospect_profile"]
        self.assertNotIn("billing_qualification", profile)
        self.assertEqual(profile["name"], "Alex Smith")
        self.assertEqual(profile["tech_stack"], ["AWS"])

    def test_cached_profile_omits_sensitive_fields(self):
        with patch.object(
            gtm_agent.data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": {**self.record, "prospect_id": "LEAD-12345"}},
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-12345")

        profile = result["prospect_profile"]
        self.assertNotIn("billing_qualification", profile)
        self.assertEqual(profile["name"], "Alex Smith")

    def test_score_prompt_omits_sensitive_fields(self):
        profile = {**self.record, "prospect_id": "LEAD-12345", "tech_stack": ["AWS"]}
        offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 10000000,
            "description": "Test offering",
        }
        response = SimpleNamespace(model_dump=lambda: {"score": 90})
        with (
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=["AWS"]),
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = response
            gtm_agent.score_prospect.func(profile, offering)

        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertNotIn("billing_qualification", prompt)
        self.assertNotIn("123-45-6789", prompt)
        self.assertNotIn("1980-01-01", prompt)
        self.assertNotIn("4111111111111111", prompt)


if __name__ == "__main__":
    unittest.main()
