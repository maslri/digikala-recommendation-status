"""Run manually after best_transformer_encoder is attached."""

import math
import os
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))


MODEL_DIR = Path(
    os.environ.get(
        "DIGIKALA_TRANSFORMER_MODEL_DIR",
        PACKAGE_ROOT / "model" / "best_transformer_encoder",
    )
)


class RuntimeSmokeTests(unittest.TestCase):
    @unittest.skipUnless(MODEL_DIR.exists(), "model artifact is not attached locally")
    def test_runtime_contract(self):
        from recommendation_prediction import RecommendationPredictor

        predictor = RecommendationPredictor(MODEL_DIR, device="cpu")
        result = predictor.predict_one(
            title="\u0639\u0627\u0644\u06cc",
            body="\u0627\u0632 \u062e\u0631\u06cc\u062f \u0631\u0627\u0636\u06cc \u0647\u0633\u062a\u0645",
        )
        self.assertIn(result["label"], {"recommended", "not_recommended", "no_idea"})
        self.assertEqual(result["source"], "model_prediction")
        self.assertIs(result["scores_are_calibrated_probabilities"], False)
        self.assertTrue(all(math.isfinite(value) for value in result["scores"].values()))
        self.assertLess(abs(sum(result["scores"].values()) - 1.0), 1e-5)


if __name__ == "__main__":
    unittest.main()
