import uuid
import os


def get_fp(base: str, suffix: str = ".png") -> str:
    return os.path.join(base, f"{uuid.uuid4().hex}{suffix}")
