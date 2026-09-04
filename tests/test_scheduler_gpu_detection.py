# SPDX-License-Identifier: Apache-2.0
"""GPU detection in the local scheduler.

These live outside tests/test_local_scheduler.py on purpose: that module is
skipped wholesale in CI ("unexpected behavior on GCP CI machines"), which is
why a scheduler that saw only one GPU on ROCm shipped unnoticed. Everything
here is pure logic with the hardware stubbed, so it runs anywhere.
"""

import pytest

import areal.infra.scheduler.local as local_mod
from areal.infra.scheduler.local import LocalScheduler, _get_device_count_safely


class _FakePlatform:
    """Stand-in for current_platform with a controllable device count."""

    device_control_env_var = "CUDA_VISIBLE_DEVICES"

    def __init__(self, count=8, raises=False):
        self._count = count
        self._raises = raises
        self.device_count_calls = 0

    def device_count(self):
        self.device_count_calls += 1
        if self._raises:
            raise RuntimeError("driver unavailable")
        return self._count


@pytest.fixture
def no_dev_nodes(monkeypatch):
    """A /dev with no nvidia*/davinci* entries, as on a ROCm host."""
    monkeypatch.setattr(local_mod.os.path, "exists", lambda p: p == "/dev")
    monkeypatch.setattr(local_mod.os, "listdir", lambda p: ["kfd", "dri", "null"])


@pytest.fixture
def platform(monkeypatch):
    def _install(**kwargs):
        fake = _FakePlatform(**kwargs)
        monkeypatch.setattr(local_mod, "current_platform", fake)
        return fake

    return _install


class TestDeviceCount:
    def test_falls_back_to_platform_when_dev_has_no_gpu_nodes(
        self, no_dev_nodes, platform
    ):
        # The ROCm case: /dev/kfd + /dev/dri, no per-device nodes to count.
        fake = platform(count=8)
        assert _get_device_count_safely() == 8
        assert fake.device_count_calls == 1

    def test_nvidia_dev_nodes_still_win(self, monkeypatch, platform):
        # Regression guard: the NVIDIA fast path must not start calling torch.
        monkeypatch.setattr(local_mod.os.path, "exists", lambda p: p == "/dev")
        monkeypatch.setattr(
            local_mod.os, "listdir", lambda p: ["nvidia0", "nvidia1", "nvidiactl"]
        )
        fake = platform(count=99)
        # "nvidiactl" has a non-digit suffix and must not be counted.
        assert _get_device_count_safely() == 2
        assert fake.device_count_calls == 0

    def test_ascend_dev_nodes_still_win(self, monkeypatch, platform):
        monkeypatch.setattr(local_mod.os.path, "exists", lambda p: p == "/dev")
        monkeypatch.setattr(local_mod.os, "listdir", lambda p: ["davinci0", "davinci1"])
        fake = platform(count=99)
        assert _get_device_count_safely() == 2
        assert fake.device_count_calls == 0

    def test_no_accelerator_reports_none(self, no_dev_nodes, platform):
        # CPU-only hosts keep their previous behavior rather than getting [].
        platform(count=0)
        assert _get_device_count_safely() is None

    def test_platform_error_reports_none(self, no_dev_nodes, platform):
        platform(raises=True)
        assert _get_device_count_safely() is None


class TestDetectGpus:
    def test_sees_every_gpu(self, no_dev_nodes, platform):
        # The bug: this returned [0] on ROCm, so all ranks shared one GPU and
        # NCCL aborted with "Duplicate GPU detected".
        platform(count=8)
        assert LocalScheduler._detect_gpus(None) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_visible_mask_takes_precedence(self, no_dev_nodes, platform, monkeypatch):
        fake = platform(count=8)
        monkeypatch.setenv(fake.device_control_env_var, "2,5")
        assert LocalScheduler._detect_gpus(None) == [2, 5]
        # The mask is authoritative; no need to ask the driver at all.
        assert fake.device_count_calls == 0

    def test_malformed_mask_falls_back_to_zero(
        self, no_dev_nodes, platform, monkeypatch
    ):
        fake = platform(count=8)
        monkeypatch.setenv(fake.device_control_env_var, "GPU-abc,GPU-def")
        assert LocalScheduler._detect_gpus(None) == [0]

    def test_cpu_only_host_falls_back_to_zero(self, no_dev_nodes, platform):
        platform(count=0)
        assert LocalScheduler._detect_gpus(None) == [0]


class _AllocStub:
    """Minimal object exposing just what _allocate_gpus touches."""

    _allocate_gpus = LocalScheduler._allocate_gpus

    def __init__(self, gpu_devices):
        self.gpu_devices = gpu_devices
        self._gpu_counter = 0


class TestGpuAllocation:
    def test_consecutive_workers_get_distinct_gpus(self):
        # This is the failure that killed training: world_size=4 workers each
        # allocating one GPU must not all land on device 0.
        sched = _AllocStub([0, 1, 2, 3, 4, 5, 6, 7])
        allocated = [sched._allocate_gpus(1)[0] for _ in range(4)]
        assert allocated == [0, 1, 2, 3]
        assert len(set(allocated)) == 4

    def test_single_device_list_reproduces_the_old_collision(self):
        # Documents the mechanism: with gpu_devices=[0] the round-robin hands
        # the same device to every worker. Detection is what must prevent this.
        sched = _AllocStub([0])
        allocated = [sched._allocate_gpus(1)[0] for _ in range(4)]
        assert allocated == [0, 0, 0, 0]

    def test_requesting_more_than_available_raises(self):
        sched = _AllocStub([0, 1])
        with pytest.raises(local_mod.GPUAllocationError):
            sched._allocate_gpus(4)

    def test_zero_gpus_is_allowed(self):
        sched = _AllocStub([0, 1])
        assert sched._allocate_gpus(0) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
