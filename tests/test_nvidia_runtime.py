"""Provisioning acceptance without network downloads, Wine, or GPU allocation."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import zipfile

import h3_nvidia_runtime as runtime
from h3_app import nvidia_postprocess as service


class RuntimeInstallTests(unittest.TestCase):
    def helper_archive(self, path, separator="/"):
        with zipfile.ZipFile(path, "w") as bundle:
            for name in runtime.HELPER_FILES:
                bundle.writestr("out" + separator + name, b"MZ fixture " + name.encode())
            bundle.writestr("README.md", "Upstream terms and instructions")
        return runtime.sha256(path)

    def test_download_rejects_mismatch_without_publishing(self):
        with TemporaryDirectory() as directory, patch.object(runtime, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b"wrong")), patch.object(runtime.time, "sleep"):
            target = Path(directory) / "bundle.zip"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                runtime.download("https://example.invalid/bundle.zip", target, "0" * 64)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".zip.partial").exists())

    def test_download_retries_then_caches_only_verified_content(self):
        with TemporaryDirectory() as directory, patch.object(runtime, "urlopen", side_effect=[OSError("interrupted"), io.BytesIO(b"verified")]) as fetch, patch.object(runtime.time, "sleep"):
            target = Path(directory) / "bundle.zip"
            checksum = hashlib.sha256(b"verified").hexdigest()
            runtime.download("https://example.invalid/bundle.zip", target, checksum)
            runtime.download("https://example.invalid/bundle.zip", target, checksum)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(target.read_bytes(), b"verified")

    def test_helper_install_is_idempotent_and_repairs_missing_libraries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.zip"
            checksum = self.helper_archive(archive)
            with patch.object(runtime, "download", return_value=archive) as fetch:
                target = runtime.install_archive(root, "helper-v1", "https://example.invalid/helper.zip", checksum, runtime.HELPER_FILES, helper=True)
                self.assertFalse((target / "ffmpeg.exe").exists())
                self.assertTrue((target / "README.md").is_file())
                runtime.install_archive(root, "helper-v1", "https://example.invalid/helper.zip", checksum, runtime.HELPER_FILES, helper=True)
                fetch.assert_called_once()
                (target / "nvngx_dlssnr.dll").unlink()
                self.helper_archive(archive)
                runtime.install_archive(root, "helper-v1", "https://example.invalid/helper.zip", checksum, runtime.HELPER_FILES, helper=True)
                self.assertEqual(fetch.call_count, 2)
                self.assertTrue((target / "nvngx_dlssnr.dll").is_file())

    def test_helper_accepts_windows_archive_paths_on_every_platform(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "windows.zip"
            checksum = self.helper_archive(archive, separator="\\")
            # Simulate Linux ZipInfo parsing even when this test runs on Windows.
            with patch.object(runtime.zipfile.os, "sep", "/"), patch.object(runtime, "download", return_value=archive):
                target = runtime.install_archive(root, "helper-v1", "https://example.invalid/helper.zip", checksum, runtime.HELPER_FILES, helper=True)
            for name in runtime.HELPER_FILES:
                self.assertEqual((target / name).read_bytes(), b"MZ fixture " + name.encode())

    def test_tar_traversal_is_rejected_before_installation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                entry = tarfile.TarInfo("../escaped")
                entry.size = 1
                bundle.addfile(entry, io.BytesIO(b"x"))
            with patch.object(runtime, "download", return_value=archive):
                with self.assertRaises(tarfile.TarError):
                    runtime.install_archive(root, "runner", "https://example.invalid/runner.tar.gz", "fixture", ("bin/wine",))
            self.assertFalse((root / "runner").exists())
            self.assertFalse((root.parent / "escaped").exists())

    def test_custom_configuration_never_inherits_managed_paths_or_error(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"H3_DLSSNR_EXE": "/custom/helper.exe"}, clear=True):
            root = Path(directory)
            runtime.write_manifest(root, {"env": {"H3_DLSSNR_EXE": "/managed/helper.exe", "H3_DLSSNR_DLL_DIR": "/managed/models"}, "error": "old failure"})
            self.assertEqual(runtime.runtime_env(root), {"H3_DLSSNR_EXE": "/custom/helper.exe"})
            with patch.object(runtime, "download") as fetch:
                runtime.ensure_runtime(root)
                fetch.assert_not_called()

    def test_failed_setup_is_visible_and_disabled_setup_does_not_download(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"H3_NVIDIA_RUNTIME_DIR": directory}, clear=True), patch.object(runtime.platform, "machine", return_value="AMD64"):
            with patch.object(runtime, "install_rh_host", side_effect=RuntimeError("network failed")), patch.object(runtime, "install_linux_packages"), patch.object(runtime, "ensure_display", return_value={}), patch.object(runtime, "check_vulkan"):
                runtime.prepare_runtime()
            self.assertIn("network failed", runtime.runtime_env()["H3_DLSSNR_SETUP_ERROR"])
            with self.assertRaisesRegex(service.H3Error, "network failed"):
                service.dlss_config()
            with patch.dict(os.environ, {"H3_ENABLE_DLSSNR": "0"}), patch.object(runtime, "download") as fetch:
                runtime.ensure_runtime()
                fetch.assert_not_called()


class PrefixStateTests(unittest.TestCase):
    def test_ngx_programdata_is_reset_without_touching_the_rest_of_the_prefix(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory)
            models = prefix / "drive_c/ProgramData/NVIDIA/NGX/models"
            models.mkdir(parents=True)
            (models / "state").write_text("stale")
            (prefix / "system.reg").write_text("registry")
            runtime.reset_ngx_state(prefix)
            runtime.reset_ngx_state(prefix)  # Absent folder is fine.
            self.assertFalse((prefix / "drive_c/ProgramData/NVIDIA/NGX").exists())
            self.assertTrue((prefix / "drive_c/ProgramData/NVIDIA").is_dir())
            self.assertEqual((prefix / "system.reg").read_text(), "registry")

    def test_wineboot_waits_for_wineserver_to_save_the_registry(self):
        with patch.object(runtime.subprocess, "run", return_value=Mock(returncode=0)) as run:
            runtime.wait_wine_server("/runner/bin/wine", {})
        args = run.call_args.args[0]
        self.assertEqual(Path(args[0]).name, "wineserver")
        self.assertEqual(args[1], "-w")
        self.assertIs(run.call_args.kwargs["stdout"], runtime.subprocess.DEVNULL)


class R580CoreTests(unittest.TestCase):
    def test_r580_core_mirrors_host_and_falls_back_across_driver_versions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host"
            (host / "caller").mkdir(parents=True)
            for name in ("dlss5nr_host.exe", "dlss5nr_bridge.dll", "nvngx_dlss.dll", "_nvngx.dll",
                         "caller/nvngx.dll_comfy.dll"):
                (host / name).write_bytes(b"MZ " + name.encode())
            attempts = []

            def fetch(url, path, expected=None):
                attempts.append(url)
                if len(attempts) == 1:
                    raise RuntimeError("Download failed: HTTP Error 404")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"driver")
                return path

            with patch.dict(os.environ, {"H3_DLSSNR_R580_DRIVER": ""}), \
                    patch.object(runtime, "runtime_env", return_value={"H3_DLSSNR_DLL_DIR": str(host)}), \
                    patch.object(runtime, "download", side_effect=fetch), \
                    patch.object(runtime, "extract_ngx_core",
                                 side_effect=lambda root, archive, core: core.write_bytes(b"MZ r580")):
                info = runtime.ensure_r580_core(root)
                again = runtime.ensure_r580_core(root)
            target = runtime.r580_host_dir(root)
            self.assertEqual(info["driver"], runtime.R580_DRIVER_CANDIDATES[1])
            self.assertEqual(again, info)
            self.assertEqual(len(attempts), 2)  # Cached after the first success.
            self.assertEqual((target / "_nvngx.dll").read_bytes(), b"MZ r580")
            self.assertEqual((target / "caller/nvngx.dll_comfy.dll").read_bytes(), b"MZ caller/nvngx.dll_comfy.dll")
            self.assertEqual((host / "_nvngx.dll").read_bytes(), b"MZ _nvngx.dll")  # Default core untouched.


class LinuxDlssSetupTests(unittest.TestCase):
    def test_wine_environment_removes_cuda_paths_and_enables_neural_rendering(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/cuda/lib64", "KEEP": "value"}, clear=True):
            env = runtime.wine_env("/prefix", "/runner/bin/wine")
        self.assertNotIn("LD_LIBRARY_PATH", env)
        self.assertEqual(env["KEEP"], "value")
        self.assertEqual(env["DXVK_NVAPI_DRS_NGX_DLSS_NR_OVERRIDE"], "on")
        self.assertEqual(env["DXVK_LOG_LEVEL"], "none")
        self.assertEqual(env["VKD3D_DEBUG"], "none")

    def test_cuda_assignment_is_translated_to_vulkan_uuid(self):
        env = {"CUDA_VISIBLE_DEVICES": "2"}
        result = Mock(stdout="0, GPU-00000000-0000-0000-0000-000000000000\n2, GPU-12345678-abcd-abcd-abcd-123456789012\n")
        with patch.object(runtime.subprocess, "run", return_value=result):
            runtime.pin_gpu(env)
        self.assertEqual(env["DXVK_FILTER_DEVICE_UUID"], "12345678abcdabcdabcd123456789012")
        self.assertEqual(env["DLSS5NR_GPU_INDEX"], "0")

    def test_vulkan_rejects_software_renderer_and_reports_driver_layer(self):
        with patch.object(runtime.subprocess, "run", return_value=Mock(returncode=0, stdout="deviceName = llvmpipe", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA Vulkan device"):
                runtime.check_vulkan({"DISPLAY": ":99"})

    def test_missing_icd_manifest_is_generated_for_mounted_driver(self):
        with TemporaryDirectory() as directory:
            library = Path(directory) / "driver" / "libGLX_nvidia.so.0"
            library.parent.mkdir()
            library.touch()
            with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(runtime, "nvidia_icd_manifests", return_value=[]),                     patch.object(runtime, "find_driver_library", return_value=(library, False)):
                env = runtime.vulkan_driver_env({}, Path(directory) / "root")
            manifest = json.loads(Path(env["VK_DRIVER_FILES"]).read_text())
            self.assertEqual(manifest["ICD"]["library_path"], str(library))
            self.assertEqual(env["VK_ICD_FILENAMES"], env["VK_DRIVER_FILES"])
            self.assertEqual(env["LD_LIBRARY_PATH"], str(library.parent))

    def test_existing_icd_or_absent_driver_leaves_environment_unchanged(self):
        with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(runtime, "nvidia_icd_manifests", return_value=[Path("/etc/vulkan/icd.d/nvidia_icd.json")]):
            self.assertEqual(runtime.vulkan_driver_env({}), {})
        with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(runtime, "nvidia_icd_manifests", return_value=[]),                 patch.object(runtime, "find_driver_library", return_value=(None, False)):
            self.assertEqual(runtime.vulkan_driver_env({}), {})

    def test_ldconfig_cache_lookup_parses_driver_path(self):
        listing = "\tlibGLX_nvidia.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0\n"
        self.assertEqual(runtime.find_driver_library("libGLX_nvidia.so.0", listing),
                         (Path("/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"), True))

    def test_wineboot_output_never_uses_pipes_held_by_wine_services(self):
        with patch.object(runtime, "ensure_display", side_effect=lambda env: env),                 patch.object(runtime.subprocess, "run", return_value=Mock(returncode=0)) as run:
            runtime.wine_run("/runner/bin/wine", ["wineboot.exe", "-u"], {"WINEDLLOVERRIDES": "dxgi=n,b"})
        kwargs = run.call_args.kwargs
        self.assertNotIn("capture_output", kwargs)
        self.assertIs(kwargs["stdin"], runtime.subprocess.DEVNULL)
        self.assertNotEqual(kwargs["stdout"], runtime.subprocess.PIPE)
        self.assertIn("mscoree,mshtml=", kwargs["env"]["WINEDLLOVERRIDES"])

    def test_vulkan_accepts_real_nvidia_device(self):
        result = Mock(returncode=0, stdout="deviceName = NVIDIA RTX PRO 6000 Blackwell", stderr="")
        with patch.object(runtime.subprocess, "run", return_value=result) as run:
            runtime.check_vulkan({"DISPLAY": ":99", "KEEP": "1"})
        # Surface/X availability must not decide whether the GPU driver works.
        self.assertNotIn("DISPLAY", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.kwargs["env"]["KEEP"], "1")

    def test_display_number_is_read_through_newline_before_pipe_closes(self):
        # Xvfb writes digits and newline separately; closing early kills it.
        fake_select = Mock(select=Mock(side_effect=lambda r, w, x, t: (r, w, x)))
        with patch.dict("sys.modules", {"select": fake_select}), \
                patch.object(runtime.os, "read", side_effect=[b"12", b"\n"]) as read:
            self.assertEqual(runtime.read_display_number(5, Mock()), "12")
        self.assertEqual(read.call_count, 2)

    def test_reachable_display_is_reused(self):
        with patch.object(runtime.os, "name", "posix"), patch.object(runtime, "display_reachable", return_value=True),                 patch.object(runtime.subprocess, "Popen") as popen:
            self.assertEqual(runtime.ensure_display({"DISPLAY": ":5"}), {"DISPLAY": ":5"})
        popen.assert_not_called()

    def test_unreachable_display_is_replaced_with_managed_xvfb(self):
        with patch.object(runtime.os, "name", "posix"), patch.object(runtime, "display_reachable", return_value=False),                 patch.object(runtime.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requires Xvfb"):
                runtime.ensure_display({"DISPLAY": ":5"})

    def test_host_download_is_checked_against_pinned_git_blob(self):
        with TemporaryDirectory() as directory:
            data = b"MZ fixture"
            expected = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            def fetch(url, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            with patch.object(runtime, "RH_DLSS_FILES", {"native/bin/dlss5nr_host.exe": expected}), patch.object(runtime, "download", side_effect=fetch) as download:
                host = runtime.install_rh_host(Path(directory))
                runtime.install_rh_host(Path(directory))
                download.assert_called_once()
                self.assertEqual((host / "dlss5nr_host.exe").read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
