"""Push the committed dataset JSONs to LangSmith. Idempotent: an existing
dataset is left alone, so experiments stay comparable across runs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client  # noqa: E402

DATASETS = Path(__file__).parent / "datasets"

if __name__ == "__main__":
    client = Client()
    for path in sorted(DATASETS.glob("*.json")):
        spec = json.loads(path.read_text())
        name = spec["dataset"]
        if client.has_dataset(dataset_name=name):
            print(f"{name}: exists, leaving as-is")
            continue
        ds = client.create_dataset(dataset_name=name, description=spec["description"])
        client.create_examples(
            dataset_id=ds.id,
            examples=[{"inputs": e["inputs"], "outputs": e["outputs"],
                       "metadata": e.get("metadata")} for e in spec["examples"]],
        )
        print(f"{name}: created with {len(spec['examples'])} examples")
