from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def save_reports(results: Dict, output_dir: Path, top_k: int = 10) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "plagiarism_report.json"
    csv_path = output_dir / "plagiarism_report.csv"
    txt_path = output_dir / "plagiarism_summary.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pairs = results["pairs"][:top_k]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_a", "doc_b", "tfidf_cosine", "fuzzy_score", "combined_score", "flagged", "shared_snippets"])
        for pair in pairs:
            writer.writerow([
                pair["doc_a"],
                pair["doc_b"],
                pair["tfidf_cosine"],
                pair["fuzzy_score"],
                pair["combined_score"],
                pair["flagged"],
                " | ".join(pair["shared_snippets"]),
            ])

    lines = []
    lines.append("PLAGIARISM DETECTION SUMMARY")
    lines.append("=" * 32)
    lines.append(f"Threshold: {results['threshold']}")
    lines.append(f"Documents analyzed: {len(results['documents'])}")
    lines.append(f"Flagged pairs: {len(results['flagged_pairs'])}")
    lines.append("")

    if results["flagged_pairs"]:
        lines.append("Flagged pairs:")
        for pair in results["flagged_pairs"]:
            lines.append(
                f"- {pair['doc_a']} <-> {pair['doc_b']} "
                f"(combined={pair['combined_score']}, tfidf={pair['tfidf_cosine']}, fuzzy={pair['fuzzy_score']})"
            )
            if pair["shared_snippets"]:
                for snippet in pair["shared_snippets"]:
                    lines.append(f"  snippet: {snippet}")
    else:
        lines.append("No pairs crossed the plagiarism threshold.")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
