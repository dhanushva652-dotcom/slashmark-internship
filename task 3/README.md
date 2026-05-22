# AI Plagiarism Checker

A classic NLP plagiarism-detection pipeline that:
- normalizes text
- extracts TF-IDF n-gram features
- computes cosine similarity across documents
- adds fuzzy matching with RapidFuzz
- applies thresholding to flag likely plagiarism
- generates JSON, CSV, and text reports

## What it supports
- `.txt` files in a folder
- pairwise similarity scoring
- configurable thresholds
- sentence-level overlap snippets for reporting

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py --input_dir samples --output_dir outputs --threshold 0.65
```

## Output
The script writes:
- `outputs/plagiarism_report.json`
- `outputs/plagiarism_report.csv`
- `outputs/plagiarism_summary.txt`

## Notes
This is a rule-based similarity detector, not a deep-learning classifier.
It is well suited for classroom plagiarism screening, newsroom deduplication, and SEO duplicate-content checks.
