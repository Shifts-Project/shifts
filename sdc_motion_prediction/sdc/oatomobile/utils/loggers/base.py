# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base logger, borrowed from DeepMind"s Acme."""

import abc
from typing import Any
from typing import Mapping

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
  """A logger has a `write` method."""

  @abc.abstractmethod
  def write(self, data: LoggingData):
    """Writes `data` to destination (file, terminal, database, etc)."""


class NoOpLogger(Logger):
  """Simple Logger which does nothing and outputs no logs.

  This should be used sparingly, but it can prove useful if we want to
  quiet an individual component and have it produce no logging
  whatsoever.
  """

  def write(self, data: LoggingData):
    pass
