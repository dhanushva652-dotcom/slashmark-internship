import argparse
from pathlib import Path

from plagiarism_detector import PlagiarismDetector
from report import save_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="AI plagiarism checker using TF-IDF, cosine similarity and fuzzy matching.")
    parser.add_argument("--input_dir", type=str, default="samples", help="Folder containing .txt files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder where reports will be saved")
    parser.add_argument("--threshold", type=float, default=0.65, help="Similarity threshold for flagging plagiarism")
    parser.add_argument("--top_k", type=int, default=10, help="Maximum number of pairs to include in the report")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    detector = PlagiarismDetector(threshold=args.threshold)
    documents = detector.load_documents(input_dir)

    if len(documents) < 2:
        raise ValueError("Need at least two .txt files in the input folder.")

    results = detector.analyze(documents)
    save_reports(results, output_dir=output_dir, top_k=args.top_k)

    print(f"Analyzed {len(documents)} documents.")
    print(f"Flagged {len(results['flagged_pairs'])} likely plagiarism pairs.")
    print(f"Reports saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
