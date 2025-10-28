import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


FILE_NAME_PREFIX = "Ses"
TRANSCRIPT_CSV_NAME = "iemocap_transcriptions.csv"
LABEL_CSV_NAME = "iemocap_labels.csv"
CLI_DESCRIPTION = "Build IEMOCAP transcription and label CSVs."
TEXT_FOLDER_ARG = "--text-folder-path"
LABEL_FOLDER_ARG = "--label-folder-path"
TEXT_OUTPUT_ARG = "--text-output-path"
LABEL_OUTPUT_ARG = "--label-output-path"

def _iter_transcriptions(path: Path) -> Iterable[tuple[str, str]]:
	"""Yield (file_name, transcription) tuples for every line in the corpus."""

	for txt_file in sorted(path.glob("*.txt")):
		for raw_line in txt_file.read_text(encoding="utf-8").splitlines():
			line = raw_line.strip()

			if not line:
				continue

			name_part, _, transcription_part = line.partition(":")

			if not name_part.startswith(FILE_NAME_PREFIX):
				continue

			if not transcription_part:
				continue

			file_name = name_part.split(" ", 1)[0].strip()
			transcription = transcription_part.strip()

			if file_name and transcription:
				yield file_name, transcription


def _iter_labels(path: Path) -> Iterable[tuple[str, str]]:
	"""Yield (file_name, emotion) tuples for every label line in the corpus."""

	pattern = re.compile(r"\s*\[[^]]+\]\s+(?P<file>\S+)\s+(?P<emotion>\S+)")
	for label_file in sorted(path.glob("*.txt")):
		lines = label_file.read_text(encoding="utf-8").splitlines()
		for line in lines[1:]:
			match = pattern.match(line)
			if not match:
				continue
			yield match.group("file"), match.group("emotion")


def _resolve_output_path(base_path: Path, output_path: str | None, default_name: str) -> Path:
	"""Return the CSV destination path, creating parent directories when needed."""

	resolved = Path(output_path) if output_path else base_path
	if resolved.is_dir():
		resolved = resolved / default_name
	resolved.parent.mkdir(parents=True, exist_ok=True)
	return resolved


def build_transcription_dataframe(folder_path: str, output_path: str | None = None) -> pd.DataFrame:
	"""
	Construct a dataframe with file names and transcriptions from IEMOCAP.

	Parameters
	----------
	folder_path:
		Directory containing the IEMOCAP transcription text files.
	output_path:
		Optional CSV file path; defaults to the folder path.
	"""

	base_path = Path(folder_path)
	if not base_path.exists():
		raise FileNotFoundError(f"Folder not found: {folder_path}")

	resolved_output = _resolve_output_path(base_path, output_path, TRANSCRIPT_CSV_NAME)
	rows = list(_iter_transcriptions(base_path))
	df = pd.DataFrame(rows, columns=["file_name", "transcription"])
	df.to_csv(resolved_output, index=False)
	return df


def build_label_dataframe(folder_path: str, output_path: str | None = None) -> pd.DataFrame:
	"""Construct a dataframe with file names and emotions from IEMOCAP labels.

	Parameters
	----------
	folder_path:
		Directory containing the IEMOCAP label text files.
	output_path:
		Optional CSV file path; defaults to the folder path.
	"""

	base_path = Path(folder_path)
	if not base_path.exists():
		raise FileNotFoundError(f"Folder not found: {folder_path}")

	resolved_output = _resolve_output_path(base_path, output_path, LABEL_CSV_NAME)
	rows = list(_iter_labels(base_path))
	df = pd.DataFrame(rows, columns=["file_name", "label"])
	df.to_csv(resolved_output, index=False)
	return df


if __name__ == "__main__":
	def _parse_args() -> argparse.Namespace:
		parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
		parser.add_argument(TEXT_FOLDER_ARG, required=True, dest="text_folder_path")
		parser.add_argument(LABEL_FOLDER_ARG, required=True, dest="label_folder_path")
		parser.add_argument(TEXT_OUTPUT_ARG, required=False, dest="text_output_path")
		parser.add_argument(LABEL_OUTPUT_ARG, required=False, dest="label_output_path")
		return parser.parse_args()

	args = _parse_args()

	build_transcription_dataframe(args.text_folder_path, args.text_output_path)
	build_label_dataframe(args.label_folder_path, args.label_output_path)