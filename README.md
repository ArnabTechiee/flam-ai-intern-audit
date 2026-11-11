# FLAM AI Intern - The Audit

## Setup
    pip install tiktoken transformers sentencepiece protobuf regex numpy

## Corpus (FLORES-200 devtest, CC-BY-SA 4.0)
    mkdir data
    curl -L -o data/flores200.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
    tar -xzf data/flores200.tar.gz -C data

## Reproduce (run from the repo root; on Windows set PYTHONIOENCODING=utf-8 first)
    python your-submission/partA/prepare_corpus.py   # A1: build the parallel corpus
    python your-submission/partA/audit.py            # A2: one flaw at a time
    python your-submission/partA/analyze.py          # A3: 5 tokenizers x 4 denominators
    python your-submission/partA/extras.py           # corpus effect + NFC token effect
    python your-submission/partA/nfc_check.py        # which characters NFC changes
    python your-submission/partB/capacity.py         # B1-B3
    python your-submission/partB/estimates.py        # FP8 + Part C GPU predictions
    python your-submission/partC/make_pairs.py       # then rate_timer.py (interactive)

Each script's output is saved next to it as *_output.txt.

## Deliverables
- your-submission/NOTEBOOK.md, your-submission/AI_USAGE.md
- Part A: partA/CORPUS.md, partA/memo.md
- Part B: partB/answers.md
- Part C: partC/memo.md
