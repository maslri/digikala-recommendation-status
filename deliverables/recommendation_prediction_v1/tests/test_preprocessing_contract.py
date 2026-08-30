import json
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from recommendation_prediction.preprocessing import (  # noqa: E402
    ID2LABEL,
    build_model_text,
    normalize_text,
)


class PreprocessingContractTests(unittest.TestCase):
    def test_label_order_is_frozen(self):
        self.assertEqual(ID2LABEL, {0: "recommended", 1: "not_recommended", 2: "no_idea"})

    def test_persian_normalization_and_tags(self):
        self.assertEqual(
            normalize_text("  \u0643\u064a\u0641\u064a\u062a  \n \u0639\u0627\u0644\u064a "),
            "\u06a9\u06cc\u0641\u06cc\u062a \u0639\u0627\u0644\u06cc",
        )
        self.assertEqual(
            build_model_text(
                title="\u062e\u0648\u0628",
                body="\u06a9\u06cc\u0641\u064a\u062a \u0639\u0627\u0644\u06cc",
            ),
            "[TITLE] \u062e\u0648\u0628 [BODY] \u06a9\u06cc\u0641\u06cc\u062a \u0639\u0627\u0644\u06cc",
        )

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model_text(title="", body=None, advantages="nan", disadvantages="  ")

    def test_arrays_are_rejected_in_schema_v1(self):
        with self.assertRaises(TypeError):
            build_model_text(
                body="\u062e\u0648\u0628",
                advantages=["\u0627\u0631\u0632\u0627\u0646"],
            )

    def test_contract_matches_label_order(self):
        contract = json.loads(
            (PACKAGE_ROOT / "config" / "integration_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["label_id_order"],
            {"0": "recommended", "1": "not_recommended", "2": "no_idea"},
        )


if __name__ == "__main__":
    unittest.main()
