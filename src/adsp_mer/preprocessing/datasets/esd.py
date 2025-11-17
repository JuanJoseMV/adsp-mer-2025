"""Utilities for extracting ESD transcriptions and labels into CSV files."""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


TRANSCRIPT_CSV_NAME = "esd_transcriptions.csv"
LABEL_CSV_NAME = "esd_labels.csv"
CLI_DESCRIPTION = "Build ESD transcription and label CSVs."
TEXT_FOLDER_ARG = "--text-folder-path"
TRANSCRIPT_OUTPUT_ARG = "--transcript-output-path"
LABEL_OUTPUT_ARG = "--label-output-path"


def _iter_esd_entries(path: Path) -> Iterable[tuple[str, str, str]]:
	"""Yield (file_name, transcription, label) tuples for every ESD line."""

	for txt_file in sorted(path.glob("*.txt")):
		for raw_line in txt_file.read_text(encoding="utf-8").splitlines():
			line = raw_line.strip()
			if not line:
				continue
			parts = line.split("\t")
			if len(parts) < 3:
				continue
			file_name, transcription, label = (part.strip() for part in parts[:3])
			if file_name and transcription and label:
				yield file_name, transcription, label


def _resolve_output_path(base_path: Path, output_path: str | None, default_name: str) -> Path:
	"""Return the CSV destination path."""

	resolved = Path(output_path) if output_path else base_path
	if resolved.is_dir():
		resolved = resolved / default_name
	resolved.parent.mkdir(parents=True, exist_ok=True)
	return resolved


def build_transcription_dataframe(folder_path: str, output_path: str | None = None) -> pd.DataFrame:
	"""Construct a dataframe with file names and transcriptions from ESD."""

	base_path = Path(folder_path)
	if not base_path.exists():
		raise FileNotFoundError(f"Folder not found: {folder_path}")

	rows = [(file_name, transcription) for file_name, transcription, _ in _iter_esd_entries(base_path)]
	resolved_output = _resolve_output_path(base_path, output_path, TRANSCRIPT_CSV_NAME)
	df = pd.DataFrame(rows, columns=["file_name", "transcription"])
	df.to_csv(resolved_output, index=False)
	return df


def build_label_dataframe(folder_path: str, output_path: str | None = None) -> pd.DataFrame:
	"""Construct a dataframe with file names and labels from ESD."""

	base_path = Path(folder_path)
	if not base_path.exists():
		raise FileNotFoundError(f"Folder not found: {folder_path}")

	rows = [(file_name, label) for file_name, _, label in _iter_esd_entries(base_path)]
	resolved_output = _resolve_output_path(base_path, output_path, LABEL_CSV_NAME)
	df = pd.DataFrame(rows, columns=["file_name", "label"])
	df.to_csv(resolved_output, index=False)
	return df


if __name__ == "__main__":
	def _parse_args() -> argparse.Namespace:
		parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
		parser.add_argument(TEXT_FOLDER_ARG, required=True, dest="text_folder_path")
		parser.add_argument(TRANSCRIPT_OUTPUT_ARG, required=False, dest="transcript_output_path")
		parser.add_argument(LABEL_OUTPUT_ARG, required=False, dest="label_output_path")
		return parser.parse_args()

	args = _parse_args()

	build_transcription_dataframe(args.text_folder_path, args.transcript_output_path)
	build_label_dataframe(args.text_folder_path, args.label_output_path)
