import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent import data_service


def get_gtm_agent_module():
    os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    return importlib.import_module("gtm_agent.gtm_agent")


class UpdateProspectInfoTests(unittest.TestCase):
    prospect_id = "LEAD-39002"

    def setUp(self):
        self.original_tech_stack = list(data_service.PROSPECTS[self.prospect_id]["tech_stack"])
        self.original_profile = data_service._PROFILES.get(self.prospect_id)

    def tearDown(self):
        data_service.PROSPECTS[self.prospect_id]["tech_stack"] = self.original_tech_stack
        if self.original_profile is None:
            data_service._PROFILES.pop(self.prospect_id, None)
        else:
            data_service._PROFILES[self.prospect_id] = self.original_profile

    def test_update_persists_technology_for_fetch_and_profile_rebuild(self):
        stale_profile = {"prospect_id": self.prospect_id, "tech_stack": self.original_tech_stack}
        data_service.save_profile_to_db(self.prospect_id, stale_profile)

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.PROSPECTS[self.prospect_id]["tech_stack"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
        self.assertIsNone(data_service.get_profile_from_db(self.prospect_id)["prospect_profile"])
        rebuilt = get_gtm_agent_module().build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

    def test_updated_technology_is_not_missing_when_scoring(self):
        gtm_agent = get_gtm_agent_module()
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        profile = {"prospect_id": self.prospect_id, "tech_stack": self.original_tech_stack}

        class ScoringStub:
            def invoke(self, messages):
                prospect_profile = json.loads(
                    messages[1]["content"].split("\n\nProspect profile:\n", 1)[1]
                )
                justification = (
                    "Terraform is present."
                    if "Terraform" in prospect_profile["tech_stack"]
                    else "Terraform is missing."
                )
                return SimpleNamespace(
                    model_dump=lambda: {
                        "score": 90,
                        "max_score": 100,
                        "justification": justification,
                        "rubric_breakdown": {},
                    }
                )

        with patch.object(gtm_agent, "_scoring_llm", ScoringStub()):
            result = gtm_agent.score_prospect.invoke({
                "prospect_profile": profile,
                "offering": data_service.get_offering("OFFER-10004"),
            })

        self.assertNotIn("missing", result["justification"].lower())

    def test_nonexistent_prospect_keeps_not_found_response(self):
        self.assertEqual(
            data_service.update_prospect_info("LEAD-NOT-FOUND", "Terraform"),
            {"updated": False, "found": False},
        )

    def test_failed_write_does_not_report_success(self):
        class FailingRecord(dict):
            def __setitem__(self, key, value):
                raise OSError("write failed")

        record = FailingRecord({"tech_stack": list(self.original_tech_stack)})
        records = {self.prospect_id: record}

        with patch.object(data_service, "PROSPECTS", records):
            result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertEqual(result, {
            "updated": False,
            "found": True,
            "tech_stack": self.original_tech_stack,
        })


if __name__ == "__main__":
    unittest.main()
