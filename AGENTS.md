# Repository Guide

## Current State

This repository contains a small Python CLI in `music_recognizer.py`. It matches audio against a local reference library and prints the associated jianpu text; it does not transcribe arbitrary audio into sheet music. Dependencies are listed in `requirements.txt` (`numpy` and `soundfile`).

## Making Changes

- Keep changes focused on the requested behavior and follow conventions established by nearby files.
- Preserve existing public interfaces unless a change explicitly requires otherwise.
- Keep the runtime CPU-only and avoid adding model-training requirements to the offline matcher.
- Run `python -m py_compile music_recognizer.py` and check `python music_recognizer.py --help` after changing the CLI.
- Add focused tests if an automated test suite is introduced; currently, validate behavior with local reference and query audio pairs.
- Update `README.md` when setup or usage changes.