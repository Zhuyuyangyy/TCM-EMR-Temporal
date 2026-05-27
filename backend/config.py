from dataclasses import dataclass

@dataclass
class Settings:
    app_name: str = "TCM-EMR-Temporal"
    version: str = "0.1.0"
    port: int = 8020

settings = Settings()
