import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingestion.pdf_extractor import main


if __name__ == "__main__":
    main()