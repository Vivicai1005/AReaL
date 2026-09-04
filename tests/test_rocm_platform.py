# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm platform.

Detection and attribute tests stub the hardware so they run on NVIDIA CI too;
only the tests that genuinely need an AMD GPU are gated behind `gpu` +
a ROCm build check.
"""

import os
import sys

import pytest
import torch

from areal.infra.platforms import (
    CudaPlatform,
    ROCmPlatform,
    UnknownPlatform,
    _init_platform,
)
from areal.infra.platforms.rocm import _parse_cpu_list

is_rocm = getattr(torch.version, "hip", None) is not None
requires_rocm = pytest.mark.skipif(not is_rocm, reason="requires a ROCm build of torch")


class TestPlatformSelection:
    """_init_platform must pick ROCm on HIP builds and leave CUDA alone."""

    @pytest.fixture
    def fake_gpu(self, monkeypatch):
        def _apply(hip_version, device_name):
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(torch.version, "hip", hip_version)
            monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: device_name)

        return _apply

    @pytest.mark.parametrize(
        "device_name",
        ["AMD Instinct MI350X", "AMD Instinct MI300X", "Radeon RX 7900 XTX"],
    )
    def test_hip_build_selects_rocm(self, fake_gpu, device_name):
        # Detection keys off the build tag, not the marketing name, so every
        # AMD product string must land on ROCmPlatform.
        fake_gpu("7.14.60850", device_name)
        assert isinstance(_init_platform(), ROCmPlatform)

    def test_nvidia_still_selects_cuda(self, fake_gpu):
        fake_gpu(None, "NVIDIA H100 80GB HBM3")
        assert isinstance(_init_platform(), CudaPlatform)

    def test_unrecognized_non_hip_still_falls_back_to_unknown(self, fake_gpu):
        fake_gpu(None, "Some Other Accelerator")
        assert isinstance(_init_platform(), UnknownPlatform)


class TestAttributeContract:
    """Attributes other subsystems read. Cheap to assert, expensive to get wrong."""

    def test_presents_as_a_cuda_device(self):
        # ROCm torch *is* torch.cuda. Callers branch on device_type == "cuda"
        # (e.g. fsdp_engine.py), so this must not become "rocm".
        assert ROCmPlatform.device_type == "cuda"
        assert ROCmPlatform.dispatch_key == "CUDA"

    def test_rccl_registers_as_nccl(self):
        assert ROCmPlatform.communication_backend == "nccl"

    def test_device_control_env_var_matches_hardcoded_readers(self):
        # areal/engine/awex/colocate_writer.py reads CUDA_VISIBLE_DEVICES
        # directly as the ground truth for relative -> physical GPU ids.
        # Switching this to HIP_VISIBLE_DEVICES would silently break that
        # mapping in colocate mode, so pin the two together.
        from areal.engine.awex import colocate_writer

        assert ROCmPlatform.device_control_env_var == "CUDA_VISIBLE_DEVICES"
        source = open(colocate_writer.__file__).read()
        assert ROCmPlatform.device_control_env_var in source

    def test_allocator_env_var_is_the_hip_one(self):
        env = ROCmPlatform.get_custom_env_vars()
        assert "PYTORCH_HIP_ALLOC_CONF" in env
        # Setting both is contradictory; the HIP one wins on ROCm.
        assert "PYTORCH_CUDA_ALLOC_CONF" not in env

    def test_get_visible_devices_reads_the_declared_var(self, monkeypatch):
        monkeypatch.setenv(ROCmPlatform.device_control_env_var, "2,3")
        assert ROCmPlatform.get_visible_devices() == ["2", "3"]


class TestParseCpuList:
    @pytest.mark.parametrize(
        "cpulist,expected",
        [
            ("0-3", {0, 1, 2, 3}),
            ("5", {5}),
            ("0-1,4-5", {0, 1, 4, 5}),
            ("0-1, 4", {0, 1, 4}),  # sysfs sometimes pads with spaces
            ("0-191\n", set(range(192))),  # trailing newline from sysfs read
            ("", set()),
        ],
    )
    def test_parses(self, cpulist, expected):
        assert _parse_cpu_list(cpulist) == expected


class _FakeProps:
    def __init__(self, domain, bus, device):
        self.pci_domain_id = domain
        self.pci_bus_id = bus
        self.pci_device_id = device


class _FakeAmdsmi:
    """amdsmi stand-in whose handle order deliberately differs from HIP's.

    Indexing `amdsmi_get_processor_handles()` by rank would therefore pick the
    wrong GPU, which is the bug this fake is built to expose.
    """

    def __init__(self):
        # HIP order is bus 0x76, 0x06, ... ; amdsmi sorts by bus, so the
        # orders do not line up. This mirrors real MI350X hardware.
        self.handles = ["h-bus06", "h-bus16", "h-bus66", "h-bus76"]
        self.by_bdf = {
            "0000:76:00.0": "h-bus76",
            "0000:06:00.0": "h-bus06",
            "0000:66:00.0": "h-bus66",
            "0000:16:00.0": "h-bus16",
        }
        self.requested_bdf = None
        self.requested_handle = None
        self.init_calls = 0
        self.shutdown_calls = 0

    def amdsmi_init(self):
        self.init_calls += 1

    def amdsmi_shut_down(self):
        self.shutdown_calls += 1

    def amdsmi_get_processor_handles(self):
        return self.handles

    def amdsmi_get_processor_handle_from_bdf(self, bdf):
        self.requested_bdf = bdf
        return self.by_bdf[bdf]

    def amdsmi_get_gpu_topo_numa_affinity(self, handle):
        self.requested_handle = handle
        # -1 = "no affinity", which makes set_numa_affinity return before
        # touching sysfs or the real process affinity.
        return -1


class TestNumaAffinityDeviceLookup:
    """amdsmi handle order != HIP device order, so lookup must go via BDF.

    Verified on MI350X: torch device 0 is amdsmi handle 3. Indexing handles
    by rank silently binds to the wrong GPU's NUMA node.
    """

    @pytest.fixture
    def fake_amdsmi(self, monkeypatch):
        fake = _FakeAmdsmi()
        monkeypatch.setitem(sys.modules, "amdsmi", fake)
        # torch device 0 sits on bus 0x76, as on real MI350X hardware.
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: _FakeProps(0, 0x76, 0)
        )
        return fake

    def test_looks_up_by_bdf_not_by_index(self, fake_amdsmi):
        ROCmPlatform.set_numa_affinity(0)
        assert fake_amdsmi.requested_bdf == "0000:76:00.0"
        # The regression: handles[0] is "h-bus06", a different GPU entirely.
        assert fake_amdsmi.requested_handle == "h-bus76"
        assert fake_amdsmi.requested_handle != fake_amdsmi.handles[0]

    def test_bdf_is_zero_padded_hex(self, monkeypatch):
        fake = _FakeAmdsmi()
        fake.by_bdf["0001:06:02.0"] = "h-other"
        monkeypatch.setitem(sys.modules, "amdsmi", fake)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: _FakeProps(1, 0x06, 0x02)
        )
        ROCmPlatform.set_numa_affinity(0)
        assert fake.requested_bdf == "0001:06:02.0"

    def test_device_properties_queried_with_the_local_rank(self, monkeypatch):
        seen = []
        fake = _FakeAmdsmi()
        monkeypatch.setitem(sys.modules, "amdsmi", fake)

        def record(i):
            seen.append(i)
            return _FakeProps(0, 0x76, 0)

        monkeypatch.setattr(torch.cuda, "get_device_properties", record)
        ROCmPlatform.set_numa_affinity(2)
        # torch.cuda enumeration already honors the visible-device mask, so
        # the rank is passed straight through without manual translation.
        assert seen == [2]

    def test_shuts_down_amdsmi_even_on_the_early_return(self, fake_amdsmi):
        ROCmPlatform.set_numa_affinity(0)
        assert fake_amdsmi.init_calls == 1
        assert fake_amdsmi.shutdown_calls == 1


class TestNumaAffinityDegradesGracefully:
    """NUMA affinity is an optimization; it must never take down a worker."""

    def test_invalid_rank_does_not_raise(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "amdsmi", _FakeAmdsmi())
        before = os.sched_getaffinity(0)
        ROCmPlatform.set_numa_affinity(99)
        assert os.sched_getaffinity(0) == before

    def test_unknown_bdf_does_not_raise(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "amdsmi", _FakeAmdsmi())
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: _FakeProps(0, 0xAB, 0)
        )
        before = os.sched_getaffinity(0)
        ROCmPlatform.set_numa_affinity(0)  # BDF not in the table
        assert os.sched_getaffinity(0) == before

    def test_missing_amdsmi_does_not_raise(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_amdsmi(name, *args, **kwargs):
            if name == "amdsmi":
                raise ImportError("no amdsmi")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_amdsmi)
        ROCmPlatform.set_numa_affinity(0)  # must not raise


@requires_rocm
@pytest.mark.gpu
class TestOnRealRocmHardware:
    def test_current_platform_is_rocm(self):
        from areal.infra.platforms import current_platform

        assert current_platform.device_name == "AMD"
        assert current_platform.communication_backend == "nccl"

    def test_torch_cuda_delegation_works(self):
        from areal.infra.platforms import current_platform

        # Platform.__getattr__ forwards unknown attrs to torch.<device_type>.
        assert current_platform.device_count() == torch.cuda.device_count()
        assert isinstance(current_platform.current_device(), int)

    def test_overridden_methods_run(self):
        ROCmPlatform.synchronize()
        ROCmPlatform().clear_memory()
        ROCmPlatform.clear_cublas_workspaces()

    def test_visible_mask_actually_restricts_torch(self):
        # The premise of the whole scheduler fix: the declared env var must
        # really narrow the process to one device.
        import subprocess

        out = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            env={**os.environ, ROCmPlatform.device_control_env_var: "1"},
        )
        assert out.stdout.strip().splitlines()[-1] == "1", out.stderr[-1000:]

    def test_set_numa_affinity_binds_cpus(self):
        before = os.sched_getaffinity(0)
        try:
            ROCmPlatform.set_numa_affinity(0)
            assert len(os.sched_getaffinity(0)) > 0
        finally:
            os.sched_setaffinity(0, before)

    def test_every_hip_device_resolves_to_a_numa_node(self):
        # Exercises the real BDF lookup for all devices, not just rank 0.
        amdsmi = pytest.importorskip("amdsmi")
        amdsmi.amdsmi_init()
        try:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                bdf = (
                    f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:"
                    f"{p.pci_device_id:02x}.0"
                )
                handle = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
                assert amdsmi.amdsmi_get_gpu_topo_numa_affinity(handle) >= 0
        finally:
            amdsmi.amdsmi_shut_down()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
