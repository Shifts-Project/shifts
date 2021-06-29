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
"""A simple CSV logger, borrowed from DeepMind"s Acme."""

import csv
import os
import time

from absl import logging

from sdc.oatomobile.utils.loggers import base


class CSVLogger(base.Logger):
  """Standard CSV logger."""

  def __init__(
      self,
      output_file: str,
      time_delta: float = 0.0,
  ):
    """Initializes the logger.

    Args:
      output_file: The filename of the output file.
      time_delta: How often (in seconds) to write values. This can be used to
        minimize terminal spam, but is 0 by default---ie everything is written.
    """

    self._output_file = output_file
    self._time = time.time()
    self._time_delta = time_delta
    self._header_exists = False
    logging.debug("Logging to {}".format(self._output_file))

  def write(self, data: base.LoggingData):
    """Writes a `data` into a row of comma-separated values."""

    # Only log if `time_delta` seconds have passed since last logging event.
    now = time.time()
    if now - self._time < self._time_delta:
      return
    self._time = now

    # Append row to CSV.
    with open(self._output_file, mode="a") as f:
      keys = sorted(data.keys())
      writer = csv.DictWriter(f, fieldnames=keys)
      if not self._header_exists:
        # Only write the column headers once.
        writer.writeheader()
        self._header_exists = True
      writer.writerow(data)

  @property
  def output_file(self) -> str:
    return self._output_file
