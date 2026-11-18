from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "part1"))
from p1_rag import build_index, run_submission


if __name__ == "__main__":
    data = ROOT / "data" / "part1"
    artifacts = ROOT / "outputs" / "part1" / "artifacts"
    index = build_index(str(data), str(data / "kb_manifest.json"), str(artifacts))
    output = ROOT / "outputs" / "part1" / "submission_public.csv"
    result = run_submission(index, str(data / "Questions.csv"), str(output))
    print({"rows": len(result), "records": len(index.records), "output": str(output)})
