# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .models import Settings

SETTINGS = Settings(_env_file=".env")

STATIC_IMAGES_DIR = "static/images"


class KeyGenerator:
    """A class to generate keys for Redis."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def result_key(self, image_id: str) -> str:
        """A Key for identifying results of the given image."""
        return f"{self.prefix}:image:{image_id}"

    def state_key(self, image_id: str) -> str:
        """A Key for identifying validation state of the given image."""
        return f"{self.prefix}:state:{image_id}"


RedisKeys = KeyGenerator(prefix="myovision")
