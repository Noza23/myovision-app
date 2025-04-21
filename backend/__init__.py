# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .models import REDIS_KEYS, Settings

SETTINGS = Settings(_env_file=".env", _env_file_encoding="utf-8")
KEYS = REDIS_KEYS(prefix="myovision")
