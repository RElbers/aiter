# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.catalog.ops.rmsnorm.tests import rmsnorm_tests
from aiter.catalog.registry import DeviceInfo

TestRmsNorm = rmsnorm_tests.to_pytest_class(DeviceInfo.current())
