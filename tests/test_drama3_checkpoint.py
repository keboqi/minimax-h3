"""CUDA lifecycle tests use fake workers; never checkpoint real processes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from h3_app.drama3.serving import checkpoint as cp, server
from h3_app.errors import H3Error


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.members = {pid: cp.Process(pid, 10, 10, pid * 100) for pid in (10, 11, 12, 13)}
        self.controller = cp.CudaCheckpoint(Path('/helper'), 10, baseline_used=cp.GIB)
        self.workers = [self.members[pid] for pid in (11, 12, 13)]
        self.before = (31 * cp.GIB, 100 * cp.GIB)
        self.after = (cp.GIB, 100 * cp.GIB)
        for name, value in (("group_processes", self.members), ("available_ram", 100 * cp.GIB),
                            ("cuda_workers", self.workers), ("device_memory", self.before)):
            mock = patch.object(cp, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        state = patch.object(self.controller, 'expect_state')
        self.state = state.start()
        self.addCleanup(state.stop)

    def test_sleep_locks_every_worker_before_checkpoint_and_restores_in_order(self):
        with patch.object(cp, 'device_memory', side_effect=[self.before, self.after]), \
                patch.object(self.controller, 'action') as action:
            self.controller.sleep()
            self.assertTrue(self.controller.transition_started)
            self.controller.wake()
        expected = [(pid, 'lock') for pid in (11, 12, 13)]
        expected += [(pid, 'checkpoint') for pid in (11, 12, 13)]
        expected += [(pid, op) for pid in (11, 12, 13) for op in ('restore', 'unlock')]
        self.assertEqual([(c.args[0].pid, c.args[1]) for c in action.call_args_list], expected)
        self.assertEqual(self.controller.required_vram, 30 * cp.GIB)
        self.assertFalse(self.controller.transition_started)
        self.assertEqual([c.args[1] for c in self.state.call_args_list],
                         ['running'] * 3 + ['checkpointed'] * 6 + ['running'] * 3)

    def test_refuses_partial_worker_visibility_or_insufficient_ram_before_locking(self):
        for workers, ram in ((self.workers[:1], 100 * cp.GIB), (self.workers, 20 * cp.GIB)):
            with self.subTest(workers=workers), patch.object(cp, 'cuda_workers', return_value=workers), \
                    patch.object(cp, 'available_ram', return_value=ram), \
                    patch.object(self.controller, 'action') as action:
                with self.assertRaises(RuntimeError):
                    self.controller.sleep()
                action.assert_not_called()
                self.assertFalse(self.controller.transition_started)

    def test_failed_checkpoint_keeps_transition_marked_for_forced_teardown(self):
        with patch.object(self.controller, 'action', side_effect=RuntimeError('driver error')):
            with self.assertRaisesRegex(RuntimeError, 'driver error'):
                self.controller.sleep()
        self.assertTrue(self.controller.transition_started)

    def test_sleep_rejects_residual_gpu_memory(self):
        with patch.object(cp, 'device_memory', return_value=self.before), \
                patch.object(cp.time, 'monotonic', side_effect=[0, 6]), \
                patch.object(self.controller, 'action'):
            with self.assertRaisesRegex(RuntimeError, 'did not release'):
                self.controller.sleep()

    def test_wake_rejects_replaced_worker(self):
        with patch.object(cp, 'device_memory', side_effect=[self.before, self.after]), \
                patch.object(self.controller, 'action') as action:
            self.controller.sleep()
            action.reset_mock()
            changed = {**self.members, 12: cp.Process(12, 10, 10, 999)}
            with patch.object(cp, 'group_processes', return_value=changed):
                with self.assertRaisesRegex(RuntimeError, 'topology changed'):
                    self.controller.wake()
            action.assert_not_called()

    def test_action_revalidates_identity_and_uses_bounded_explicit_operation(self):
        worker = self.members[11]
        with patch.object(cp, 'process_info', return_value=worker), patch.object(cp, 'run') as run:
            self.controller.action(worker, 'lock')
            self.assertEqual(run.call_args.args[0][-2:], ['--timeout', '10000'])
            self.assertEqual(run.call_args.kwargs['timeout'], 60)
        with patch.object(cp, 'process_info', return_value=cp.Process(11, 20, 20, 99)), \
                patch.object(cp, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                self.controller.action(worker, 'restore')
            run.assert_not_called()

    def test_driver_state_failure_before_mutation_does_not_lock_workers(self):
        self.state.side_effect = RuntimeError('operation not supported')
        with patch.object(self.controller, 'action') as action:
            with self.assertRaisesRegex(RuntimeError, 'not supported'):
                self.controller.sleep()
            action.assert_not_called()
        self.assertFalse(self.controller.transition_started)

    def test_driver_success_without_checkpointed_state_is_rejected(self):
        self.state.side_effect = [None] * 3 + [RuntimeError('expected checkpointed')]
        with patch.object(self.controller, 'action'):
            with self.assertRaisesRegex(RuntimeError, 'expected checkpointed'):
                self.controller.sleep()
        self.assertTrue(self.controller.transition_started)

    def test_expect_state_uses_local_worker_pid_and_rejects_wrong_state(self):
        worker = self.workers[0]
        with patch.object(cp, 'process_info', return_value=worker), patch.object(cp, 'run', return_value='running') as run:
            cp.CudaCheckpoint.expect_state(self.controller, worker, 'running')
            self.assertEqual(run.call_args.args[0], [str(Path('/helper')), '--get-state', '--pid', '11'])
            with self.assertRaisesRegex(RuntimeError, 'expected checkpointed'):
                cp.CudaCheckpoint.expect_state(self.controller, worker, 'checkpointed')

    def test_device_counters_can_lag_checkpoint_state(self):
        with patch.object(cp, 'device_memory', side_effect=[self.before, self.before, self.after]), \
                patch.object(cp.time, 'sleep') as sleep, patch.object(self.controller, 'action'):
            self.controller.sleep()
            sleep.assert_called_once_with(0.1)

    def test_small_allocation_cannot_pass_sleep_without_releasing_memory(self):
        used = self.controller.baseline_used + cp.GIB // 2
        with patch.object(cp, 'device_memory', return_value=(used, 100 * cp.GIB)), \
                patch.object(cp.time, 'monotonic', side_effect=[0, 6]), \
                patch.object(self.controller, 'action'):
            with self.assertRaisesRegex(RuntimeError, 'did not release'):
                self.controller.sleep()


class HostTests(unittest.TestCase):
    def test_cuda_workers_come_from_group_device_handles_even_when_nvml_reports_pid_one(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            members = {pid: cp.Process(pid, 10, 10, 100) for pid in (10, 11, 12, 13)}
            # Root has only a control handle; stage workers use the physical
            # /dev/nvidia7 even though CUDA's visible ordinal is 0 on Modal.
            for pid in members:
                fd = proc / str(pid) / 'fd'
                fd.mkdir(parents=True)
                (fd / '0').touch()
            (proc / '1/fd').mkdir(parents=True)
            (proc / '1/fd/0').touch()
            def readlink(path):
                return '/dev/nvidiactl' if path.parent.parent.name == '10' else '/dev/nvidia7'
            with patch.object(cp.os, 'readlink', side_effect=readlink), patch.object(cp, 'run') as run:
                workers = cp.cuda_workers(members, proc)
                self.assertEqual([p.pid for p in workers], [11, 12, 13])
                run.assert_not_called()  # never asks NVML for worker PIDs

    def test_unreadable_worker_handles_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            (proc / '11/fd').mkdir(parents=True)
            (proc / '11/fd/0').touch()
            with patch.object(cp.os, 'readlink', side_effect=PermissionError('hidden')):
                with self.assertRaises(PermissionError):
                    cp.cuda_workers({11: cp.Process(11, 10, 10, 100)}, proc)

    def test_device_memory_is_independent_of_virtualized_nvml_pids(self):
        with patch.object(cp, 'run', return_value='5120, 97280') as run:
            self.assertEqual(cp.device_memory(), (5120 * 2**20, 97280 * 2**20))
            self.assertIn('--query-gpu=memory.used,memory.total', run.call_args.args[0])
        with patch.object(cp, 'run', return_value='100, 10'):
            with self.assertRaisesRegex(RuntimeError, 'Invalid'):
                cp.device_memory()
    def test_proc_parser_handles_spaces_parentheses_and_excludes_other_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pid, group, session, state in ((10, 10, 10, 'S'), (11, 10, 10, 'S'),
                                              (12, 20, 20, 'S'), (13, 10, 10, 'Z')):
                directory = root / str(pid)
                directory.mkdir()
                fields = [state, '1', str(group), str(session)] + ['0'] * 15 + ['123']
                (directory / 'stat').write_text(f'{pid} (worker (test)) ' + ' '.join(fields))
            self.assertEqual(set(cp.group_processes(10, root)), {10, 11})
            self.assertEqual(cp.process_info(11, root).started, 123)

    def test_ram_is_bounded_by_parent_cgroup_and_current_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proc, cgroup = root / 'proc', root / 'cgroup'
            (proc / 'self').mkdir(parents=True)
            (cgroup / 'child').mkdir(parents=True)
            (proc / 'meminfo').write_text('MemAvailable: 104857600 kB\n')
            (proc / 'self/cgroup').write_text('0::/child\n')
            (cgroup / 'memory.max').write_text(str(40 * cp.GIB))
            (cgroup / 'memory.current').write_text(str(20 * cp.GIB))
            (cgroup / 'child/memory.max').write_text('max')
            self.assertEqual(cp.available_ram(proc, cgroup), 20 * cp.GIB)

    def test_old_driver_does_not_download_helper(self):
        with patch.object(cp.platform, 'system', return_value='Linux'), \
                patch.object(cp.platform, 'machine', return_value='x86_64'), \
                patch.object(cp, 'run', return_value='610.10\n575.64'), \
                patch.object(cp.requests, 'get') as get:
            with self.assertRaisesRegex(RuntimeError, '580'):
                cp.prepare_helper(Path('/unused'))
            get.assert_not_called()

    def test_helper_download_must_match_pinned_checksum(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(cp.platform, 'system', return_value='Linux'), \
                patch.object(cp.platform, 'machine', return_value='x86_64'), \
                patch.object(cp, 'run', return_value='610.10'), \
                patch.object(cp.requests, 'get', return_value=MagicMock(content=b'wrong')):
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                cp.prepare_helper(Path(temp))
            self.assertFalse(list(Path(temp).rglob('cuda-checkpoint')))

    def test_verified_helper_is_cached_and_requires_job_support(self):
        import hashlib

        data = b'test executable'
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(cp.platform, 'system', return_value='Linux'), \
                patch.object(cp.platform, 'machine', return_value='x86_64'), \
                patch.object(cp, 'HELPER_SHA256', hashlib.sha256(data).hexdigest()), \
                patch.object(cp, 'run', side_effect=['610.10', '--launch-job --action'] * 2), \
                patch.object(cp.requests, 'get', return_value=MagicMock(content=data)) as get:
            first = cp.prepare_helper(Path(temp))
            self.assertTrue(first[1])
            self.assertEqual(cp.prepare_helper(Path(temp)), first)
            get.assert_called_once_with(cp.HELPER_URL, timeout=30)

    def test_driver_without_ipc_checkpointing_needs_no_job_support(self):
        import hashlib

        data = b'test executable'
        for versions, help_text, ipc in (('580.82.07', '--action', False),
                                         ('610.10\n609.1', '--action', False),
                                         ('610.10', '--action', None)):
            with self.subTest(versions=versions), tempfile.TemporaryDirectory() as temp, \
                    patch.object(cp.platform, 'system', return_value='Linux'), \
                    patch.object(cp.platform, 'machine', return_value='x86_64'), \
                    patch.object(cp, 'HELPER_SHA256', hashlib.sha256(data).hexdigest()), \
                    patch.object(cp, 'run', side_effect=[versions, help_text]), \
                    patch.object(cp.requests, 'get', return_value=MagicMock(content=data)):
                if ipc is None:  # 610+ is launched in a job, so the helper must offer one
                    with self.assertRaisesRegex(RuntimeError, 'job support'):
                        cp.prepare_helper(Path(temp))
                else:
                    self.assertEqual(cp.prepare_helper(Path(temp))[1], ipc)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.server = server.SglangOmniServer(server.ServerSettings(root, root, root))
        self.server.process = MagicMock(pid=10)
        self.server.process.poll.return_value = None
        self.server._checkpoint_helper = Path('/helper')
        self.controller = MagicMock(required_vram=30 * cp.GIB)
        self.controller.transition_started = True
        self.server.healthy = MagicMock(return_value=True)
        mock = patch.object(server, 'gpu_memory', return_value=None)
        mock.start()
        self.addCleanup(mock.stop)

    def test_sleep_then_wake_reuses_process_and_release_is_idempotent(self):
        process = self.server.process
        with patch.object(cp, 'CudaCheckpoint', return_value=self.controller), \
                patch.object(self.server, '_start') as start:
            self.server.release()
            self.server.release()
            self.controller.sleep.assert_called_once()
            self.assertTrue(self.server.sleeping)
            order = []
            self.controller.wake.side_effect = lambda: order.append('wake')
            self.server.ensure_ready(lambda: order.append('free_comfy'))
            self.assertEqual(order, ['free_comfy', 'wake'])
            self.assertFalse(self.server.sleeping)
            self.assertIs(self.server.process, process)
            start.assert_not_called()

    def test_no_wake_until_comfy_released_and_enough_vram(self):
        self.server._checkpoint = self.controller
        with self.assertRaises(H3Error):
            self.server.ensure_ready(MagicMock(side_effect=H3Error('busy')))
        with patch.object(server, 'gpu_memory', return_value=(1024, 100000, 'other')):
            with self.assertRaisesRegex(H3Error, 'Not enough'):
                self.server.ensure_ready(lambda: None)
        self.controller.wake.assert_not_called()
        self.assertTrue(self.server.sleeping)

    def test_preload_never_stops_the_audio_tools_worker_to_make_room(self):
        self.server.process = None
        with patch.object(server, 'gpu_memory', return_value=(1024, 100000, 'audio tools')), \
                patch.object(server, 'reclaim_audio_tools', return_value=None) as reclaim:
            with self.assertRaisesRegex(H3Error, 'needs about'):
                self.server.ensure_ready(lambda: None, evict=False)
            reclaim.assert_not_called()
            with self.assertRaisesRegex(H3Error, 'needs about'):
                self.server.ensure_ready(lambda: None)
            reclaim.assert_called_once()

    def test_sleep_failure_stops_instead_of_reusing_partial_checkpoint(self):
        self.controller.sleep.side_effect = RuntimeError('unsupported allocation')
        with patch.object(cp, 'CudaCheckpoint', return_value=self.controller), \
                patch.object(self.server, 'stop') as stop:
            self.server.release()
            stop.assert_called_once()
            self.assertTrue(self.server._sleep_disabled)

    def test_sleep_precondition_miss_stops_but_keeps_sleep_for_next_launch(self):
        self.controller.transition_started = False
        self.controller.sleep.side_effect = RuntimeError('needs more host RAM')
        with patch.object(cp, 'CudaCheckpoint', return_value=self.controller), \
                patch.object(self.server, 'stop') as stop:
            self.server.release()
            stop.assert_called_once()
            self.assertFalse(self.server._sleep_disabled)

    def test_wake_failure_restarts_once(self):
        self.server._checkpoint = self.controller
        self.controller.wake.side_effect = RuntimeError('restore failed')
        order = []
        def stop():
            order.append('stop')
            self.server.process = None
            self.server._checkpoint = None
        with patch.object(self.server, 'stop', side_effect=stop), \
                patch.object(self.server, '_start', side_effect=lambda: order.append('start')), \
                patch.object(self.server, '_wait_healthy', side_effect=lambda p: order.append('ready')):
            self.server.ensure_ready(lambda: order.append('free'))
        self.assertEqual(order, ['free', 'stop', 'start', 'ready'])
        self.assertTrue(self.server._sleep_disabled)

    def test_active_speech_prevents_sleep_without_serializing_requests(self):
        def speech(*args, **kwargs):
            self.assertEqual(self.server._active_requests, 1)
            with self.assertRaisesRegex(H3Error, 'still generating'):
                self.server.release()
            return MagicMock(ok=True, content=b'wav')
        with patch.object(self.server.http(), 'post', side_effect=speech):
            self.assertEqual(self.server.speech({}), b'wav')
        self.assertEqual(self.server._active_requests, 0)

    def test_timeout_forces_stop_and_asleep_rejects_speech(self):
        with patch.object(self.server.http(), 'post', side_effect=requests.Timeout):
            with self.assertRaises(requests.Timeout):
                self.server.speech({})
        with patch.object(self.server, 'stop') as stop, patch.object(cp, 'CudaCheckpoint') as create:
            self.server.release()
            stop.assert_called_once()
            create.assert_not_called()
        self.server._checkpoint = self.controller
        with patch.object(self.server.http(), 'post') as post:
            with self.assertRaisesRegex(H3Error, 'asleep'):
                self.server.speech({})
            post.assert_not_called()

    def test_explicit_unload_discards_host_checkpoint(self):
        with patch.object(server, 'stop_all') as stop, patch.object(server, 'release_all') as release:
            server.release_for_job('unload')
            stop.assert_called_once()
            release.assert_not_called()

    def test_start_uses_checkpoint_job_or_shm_platform_and_off_avoids_helper(self):
        from dataclasses import replace

        for mode, ipc in (('auto', True), ('auto', False), ('off', None)):
            self.server.settings = replace(self.server.settings, sleep_mode=mode)
            with self.subTest(mode=mode, ipc=ipc), \
                    patch.object(cp, 'prepare_helper', return_value=(Path('/helper'), ipc)) as prepare, \
                    patch.object(cp, 'device_memory', return_value=(cp.GIB, 100 * cp.GIB)), \
                    patch.object(server.subprocess, 'Popen') as popen, \
                    patch.object(server.atexit, 'register'):
                self.server._start()
                self.server._log.close()
                self.server._log = None
                command, env = popen.call_args.args[0], popen.call_args.kwargs['env']
                if ipc:
                    self.assertEqual(command[:2], [str(Path('/helper')), '--launch-job'])
                else:
                    self.assertEqual(command, self.server.command())
                if ipc is False:  # an older driver: stages exchange tensors through host SHM
                    self.assertEqual(env['SGLANG_OMNI_PLATFORM_SPEC'], server.SHM_PLATFORM_SPEC)
                    self.assertEqual(env['PYTHONPATH'], str(server.SHM_PLATFORM_DIR))
                    self.assertTrue((server.SHM_PLATFORM_DIR / 'h3_omni_shm.py').is_file())
                else:
                    self.assertNotIn('SGLANG_OMNI_PLATFORM_SPEC', env)
                self.assertEqual(prepare.call_count, 0 if mode == 'off' else 1)
                self.assertEqual(self.server._checkpoint_helper, None if mode == 'off' else Path('/helper'))
                self.assertTrue(popen.call_args.kwargs['start_new_session'])

    def test_snapshot_capture_starts_on_shm_without_checkpoint_job(self):
        server.SNAPSHOT_CAPTURE.set()
        self.addCleanup(server.SNAPSHOT_CAPTURE.clear)
        with patch.object(cp, 'prepare_helper') as prepare, \
                patch.object(server.subprocess, 'Popen') as popen, \
                patch.object(server.atexit, 'register'):
            self.server._start()
            self.server._log.close()
            self.server._log = None
        command, env = popen.call_args.args[0], popen.call_args.kwargs['env']
        self.assertEqual(command, self.server.command())
        self.assertEqual(env['SGLANG_OMNI_PLATFORM_SPEC'], server.SHM_PLATFORM_SPEC)
        self.assertEqual(env['PYTHONPATH'], str(server.SHM_PLATFORM_DIR))
        prepare.assert_not_called()
        self.assertIsNone(self.server._checkpoint_helper)

    def test_wrapped_start_failure_retries_once_without_checkpoint(self):
        self.server.process = None
        order = []
        def start():
            self.server._checkpoint_helper = None if self.server._sleep_disabled else Path('/helper')
            order.append(('start', self.server._sleep_disabled))
        with patch.object(self.server, '_start', side_effect=start), \
                patch.object(self.server, '_wait_healthy', side_effect=[H3Error('bad launch'), None]), \
                patch.object(self.server, 'stop', side_effect=lambda: order.append(('stop', None))):
            self.server.ensure_ready(lambda: None)
        self.assertEqual(order, [('start', False), ('stop', None), ('start', True)])

    def test_unwrapped_start_failure_is_not_retried(self):
        self.server.process = None
        self.server._sleep_disabled = True
        with patch.object(self.server, '_start') as start, \
                patch.object(self.server, '_wait_healthy', side_effect=H3Error('bad launch')), \
                patch.object(self.server, 'stop') as stop:
            with self.assertRaisesRegex(H3Error, 'bad launch'):
                self.server.ensure_ready(lambda: None)
        start.assert_called_once()
        stop.assert_called_once()

    def test_stop_kills_checkpointed_group_even_if_parent_already_exited(self):
        self.server._checkpoint = self.controller
        self.server.process.poll.return_value = 1
        with patch.object(server.os, 'name', 'posix'), \
                patch.object(server.os, 'killpg', create=True) as kill, \
                patch.object(server.signal, 'SIGKILL', 9, create=True), \
                patch.object(cp, 'group_processes', return_value={}) as members:
            self.server.stop()
        self.assertEqual(kill.call_args_list[0].args, (10, 9))
        members.assert_called_once_with(10)
        self.assertFalse(self.server.sleeping)
        self.assertIsNone(self.server.process)

    def test_stop_refuses_handoff_if_workers_survive_kill(self):
        self.server._checkpoint = self.controller
        with patch.object(server.os, 'name', 'posix'), \
                patch.object(server.os, 'killpg', create=True), \
                patch.object(server.signal, 'SIGKILL', 9, create=True), \
                patch.object(cp, 'group_processes', return_value={11: object()}), \
                patch.object(server.time, 'monotonic', side_effect=[0, 11]):
            with self.assertRaisesRegex(H3Error, 'handoff was refused'):
                self.server.stop()
        self.assertIsNotNone(self.server.process)  # retain handle for retry


class ShmPlatformTests(unittest.TestCase):
    def test_spec_resolves_to_a_cuda_platform_that_keeps_stage_traffic_off_cuda_ipc(self):
        """Stand-ins for the Omni venv's modules; the real ones exist only there."""
        import enum
        import pkgutil
        import sys
        import types

        transport = enum.Enum('TransportKind', 'CUDA_IPC SHM')
        cuda = type('CUDAOmniPlatform', (), {'get_intra_node_transport': lambda self: transport.CUDA_IPC})
        modules = {'sglang_omni': types.ModuleType('sglang_omni'),
                   'sglang_omni.comm': types.ModuleType('sglang_omni.comm'),
                   'sglang_omni.comm.data_ref': types.SimpleNamespace(TransportKind=transport),
                   'sglang_omni.platforms': types.ModuleType('sglang_omni.platforms'),
                   'sglang_omni.platforms.cuda': types.SimpleNamespace(CUDAOmniPlatform=cuda)}
        with patch.dict(sys.modules, modules), patch.object(sys, 'path', [str(server.SHM_PLATFORM_DIR), *sys.path]):
            sys.modules.pop('h3_omni_shm', None)
            try:
                platform = pkgutil.resolve_name(server.SHM_PLATFORM_SPEC)
                self.assertTrue(issubclass(platform, cuda))
                self.assertIs(platform().get_intra_node_transport(), transport.SHM)
            finally:
                sys.modules.pop('h3_omni_shm', None)


if __name__ == '__main__':
    unittest.main()
