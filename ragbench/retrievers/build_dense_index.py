from pathlib import Path
from api.dependencies import load_json
from retrievers.dense import DenseRetriever

DATA_DIR = Path("Data/beir_scifact")

def main():
    chunks = load_json(DATA_DIR / "chunks.json")
    dense = DenseRetriever(chunks)
    dense.build_index()
    print("Dense index built successfully.")

if __name__ == "__main__":
    main()