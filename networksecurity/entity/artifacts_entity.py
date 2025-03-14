from dataclasses import dataclass
from datetime import datetime
import os

@dataclass
class DataIngestionArtifact:
    training_file_path:str
    test_file_path:str
        

