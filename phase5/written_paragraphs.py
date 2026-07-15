# phase5/written_paragraphs.py
"""
Written paragraphs required by the Phase 5 brief:
  P1 — Multiple comparison correction justification
  P2 — Business recommendation narrative (generated dynamically)
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5.config import OUTPUT_DIR


CORRECTION_JUSTIFICATION = """
──────────────────────────────────────────────────────────────────────────────
WRITTEN PARAGRAPH 1 — Multiple Comparison Correction: Justification
──────────────────────────────────────────────────────────────────────────────

When running a single hypothesis test, the Type I error rate is controlled at
α = 0.05 by construction. But when we test the same treatment across three
device segments (mobile, desktop, tablet), the family-wise error rate (FWER)
inflates: the probability of at least one false positive across all three tests
is 1 − (1 − 0.05)³ ≈ 14.3%. That means we would expect a spurious "significant"
result more than 1-in-7 times even if the banner had zero true effect on any
device — an unacceptable risk when a false positive means shipping a degraded
banner experience to millions of users.

BONFERRONI vs. BENJAMINI-HOCHBERG:

Bonferroni (FWER control) divides α by the number of comparisons
(α' = 0.05 / 3 ≈ 0.0167). It guarantees that the probability of *any*
false positive in the family is ≤ 5%. This is the appropriate method for
*confirmatory* experiments where each comparison is pre-specified and a single
false positive would directly trigger a product decision — i.e., shipping a
bad experience to a real user cohort.

Benjamini-Hochberg FDR controls the *expected proportion* of discoveries that
are false (the False Discovery Rate), not the probability of any false positive.
It is more powerful (fewer false negatives) and is the right tool for
*exploratory* scans — for example, scanning 50+ secondary metrics in a single
experiment, or running a weekly automated anomaly-detection pass across hundreds
of metric × segment combinations. There, you accept that perhaps 5% of the
"significant" flags will be wrong, which is fine when a human analyst reviews
the shortlist before acting.

WHICH ONE APPLIES HERE:

This experiment — Experiment 3, discount banner placement — is confirmatory,
not exploratory. The three device segments (mobile, desktop, tablet) were
pre-specified in Phase 1 as the exact segments of interest, not discovered
post-hoc. Each significant result would directly justify shipping the new banner
to that device type, affecting millions of real users. A false positive here
does not produce a footnote in a report; it triggers an irreversible product
change. Therefore, Bonferroni correction is the appropriate and conservative
choice. BH-FDR is reported alongside for transparency and comparison,
but the decision rule is based on Bonferroni-adjusted p-values.
──────────────────────────────────────────────────────────────────────────────
"""


def generate_paragraphs(recommendation_narrative: str,
                         cost_inr: float,
                         output_path: str = None) -> str:
    """
    Combine both written paragraphs and optionally save to file.
    """
    para2_header = f"""
──────────────────────────────────────────────────────────────────────────────
WRITTEN PARAGRAPH 2 — Business Recommendation + ₹ Cost of Uncorrected Decision
──────────────────────────────────────────────────────────────────────────────

Estimated monthly cost of shipping on uncorrected p-values: ₹{cost_inr:,.0f}

{recommendation_narrative}
──────────────────────────────────────────────────────────────────────────────
"""

    full_text = CORRECTION_JUSTIFICATION + "\n" + para2_header

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"  [SAVED] Written paragraphs → {output_path}")

    return full_text


if __name__ == "__main__":
    print(CORRECTION_JUSTIFICATION)