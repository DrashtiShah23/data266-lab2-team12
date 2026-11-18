import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from p1_rag import build_index, run_submission


def find_data() -> Path:
    candidates = []
    for base in [Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input")), ROOT.parent, Path.cwd()]:
        if base.exists():
            candidates.extend(base.rglob("kb_manifest.json"))
    if not candidates:
        raise FileNotFoundError("Could not find kb_manifest.json in the Kaggle input or working directory")
    return candidates[0].parent


if __name__ == "__main__":
    data = find_data()
    questions = os.environ.get("QUESTIONS_PATH", str(data / "Questions.csv"))
    artifact_dir = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
    index = build_index(str(data), str(data / "kb_manifest.json"), str(artifact_dir))
    out = run_submission(index, questions, "submission.csv")
    print({"rows": len(out), "records": len(index.records), "questions": questions, "output": "submission.csv"})
