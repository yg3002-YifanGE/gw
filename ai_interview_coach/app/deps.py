import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "../kaggle_data")
    index_dir: str = os.getenv("INDEX_DIR", "./data/index")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


settings = Settings()

