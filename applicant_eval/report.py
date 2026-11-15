"""Self-contained HTML report; applicant text is always escaped."""

from html import escape


def render_html(report):
    queue = {q["applicant_id"]: q["reason"] for q in report.get("review_queue", [])}
    cards = []
    for r in report["results"]:
        criteria = []
        for c in r["evidence"]:
            sources = "".join(
                f"<blockquote>{escape(s['quote'])}<footer>{escape(s['evidence_id'])} · "
                f"characters {s['start']}–{s['end']} · support {s['support_score']:.3f} · "
                f"contradiction {s['contradiction_score']:.3f} · "
                f"context: {escape(', '.join(s.get('context_flags', [])) or 'no flag')}</footer></blockquote>"
                for s in c["sources"]
            )
            criteria.append(
                f"<details><summary>{escape(c['description'])} — "
                f"<b>{escape(c['status'])}</b></summary>{sources or 'No evidence provided.'}</details>"
            )
        cards.append(
            f"<article><h2>{r['rank']}. {escape(r['applicant_id'])}</h2>"
            f'<p class="tag">{escape(r["review_status"])} · '
            f"{escape(queue.get(r['applicant_id'], 'outside current review budget'))}</p>"
            f"<p>Closest profile: <b>{escape(r['closest_profile'])}</b> · "
            f"similarity {r['profile_similarity']:.3f}</p>"
            f"<p>Best-supported path: {escape(r['best_supported_profile'])}. "
            f"Unresolved: {escape(', '.join(r['unresolved_criteria']) or 'none in this rubric')}.</p>"
            + "".join(criteria)
            + "</article>"
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Applicant evidence review</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:40px auto;max-width:980px;padding:0 24px;color:#202a35;background:#f6f8fa}
h1{font-size:32px}h2{font-size:22px}article{background:white;padding:24px;margin:20px 0;border:1px solid #d7dfe6;border-radius:12px}
.tag{font-weight:600;color:#245e75}summary{cursor:pointer;padding:9px 0}blockquote{margin:12px 0;padding:12px 18px;border-left:3px solid #75a4b5;background:#f6f8fa;white-space:pre-wrap}footer{font-size:12px;color:#556373;margin-top:8px}
</style><h1>"""
        + escape(report["role_title"])
        + """ — evidence review</h1>
<p>This report prioritizes human review. It does not advance or reject applicants.
Quotes are applicant claims, not verified facts. Model scores are not probabilities of job performance.
“Unknown” means the retrieved evidence did not establish the capability; it does not mean the person lacks it.</p>
<p>All applicants remain visible below. The review queue reserves capacity for sampled lower-ranked records.</p>"""
        + "".join(cards)
        + "</html>"
    )
