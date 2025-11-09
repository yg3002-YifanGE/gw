import os
from services.retriever import TfIdfRetriever
from app.deps import settings


def main() -> None:
    retriever = TfIdfRetriever(settings.data_dir, settings.index_dir)
    stats = retriever.build_index()
    print(f"Indexed {stats['docs_indexed']} docs into {settings.index_dir}")


if __name__ == "__main__":
    main()

