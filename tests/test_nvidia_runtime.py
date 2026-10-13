"""Provisioning acceptance without network downloads, Wine, or GPU allocation."""
import ast
from contextlib import ExitStack, nullcontext
import hashlib
import gzip
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

    def test_manifest_defaults_discover_helper_without_exports(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            archive = root / "fixture.zip"
            self.helper_archive(archive)
            with patch.object(runtime, "download", return_value=archive), patch.object(runtime.platform, "system", return_value="Windows"), patch.object(runtime.platform, "machine", return_value="AMD64"):
                if os.name != "nt":
                    self.skipTest("Windows fixture exercises the native branch")
                runtime.ensure_runtime(root)
            os.environ["H3_NVIDIA_RUNTIME_DIR"] = str(root)
            exe, library, model, wine = service.dlss_config(verify=True)
            self.assertTrue(exe.is_file())
            self.assertEqual(model.parent, library)
            self.assertIsNone(wine)
            self.assertTrue(runtime.read_manifest(root)["initialized"])
            self.assertEqual(runtime.runtime_env()["H3_DLSSNR_MODEL_SHA256"], runtime.sha256(model))

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
            with patch.object(runtime, "install_archive", side_effect=RuntimeError("network failed")):
                runtime.prepare_runtime()
            self.assertIn("network failed", runtime.runtime_env()["H3_DLSSNR_SETUP_ERROR"])
            with self.assertRaisesRegex(service.H3Error, "network failed"):
                service.dlss_config()
            with patch.dict(os.environ, {"H3_ENABLE_DLSSNR": "0"}), patch.object(runtime, "download") as fetch:
                runtime.ensure_runtime()
                fetch.assert_not_called()


class LinuxRuntimeTests(unittest.TestCase):
    def test_matching_driver_bridges_are_reused_without_downloading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "drivers/610.50"
            cached.mkdir(parents=True)
            for name in runtime.NGX_FILES:
                (cached / name).write_bytes(b"bridge")
            (cached / "source.json").write_text(json.dumps({"driver_version": "610.50",
                "files": {name: runtime.sha256(cached / name) for name in runtime.NGX_FILES}}))
            with patch.object(runtime, "BRIDGE_DIRS", ()), patch.object(runtime.subprocess, "run", return_value=Mock(stdout="610.50\n")), patch.object(runtime, "download") as fetch:
                self.assertEqual(runtime.driver_bridges(root), (cached, "610.50"))
            fetch.assert_not_called()

    def test_missing_driver_components_extract_verified_package_without_installing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "nvidia.deb"
            archive.write_bytes(b"vendor archive fixture")
            checksum = runtime.sha256(archive)
            index = root / "Packages.gz"
            index.write_bytes(gzip.compress((
                "Package: libnvidia-gl-580\nArchitecture: amd64\nVersion: 580.126.20-1ubuntu1\n"
                "Filename: ./libnvidia-gl-580_580.126.20-1ubuntu1_amd64.deb\nSHA256: " + checksum
            ).encode()))

            def run(command, **kwargs):
                if command[0] == "nvidia-smi":
                    return Mock(stdout="580.126.20\n")
                self.assertEqual(command[:3], ["dpkg-deb", "-x", str(archive)])
                extracted = Path(command[-1]) / "usr/lib/x86_64-linux-gnu"
                (extracted / "nvidia/wine").mkdir(parents=True)
                (extracted / "libnvidia-ngx.so.580.126.20").write_bytes(b"native core")
                for name in ("nvngx.dll", "_nvngx.dll"):
                    (extracted / "nvidia/wine" / name).write_bytes(b"bridge")
                return Mock(returncode=0)

            with patch.object(runtime, "BRIDGE_DIRS", ()), patch.object(runtime.subprocess, "run", side_effect=run) as process, patch.object(runtime.shutil, "which", return_value="dpkg-deb"), patch.object(runtime, "download", side_effect=[index, archive]) as fetch:
                bridges, version = runtime.driver_bridges(root)
                self.assertEqual(runtime.driver_components(root, version), bridges)
            self.assertEqual(version, "580.126.20")
            self.assertEqual(process.call_count, 2)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(fetch.call_args.args[2], checksum)
            self.assertIn("/libnvidia-gl-580_580.126.20-1ubuntu1_amd64.deb", fetch.call_args.args[0])
            self.assertEqual((bridges / "libnvidia-ngx.so.1").read_bytes(), b"native core")
            self.assertTrue((bridges / "_nvngx.dll").is_file())
            self.assertTrue((bridges / "source.json").is_file())
            self.assertFalse(archive.exists())

    def test_package_resolution_requires_exact_driver_architecture_and_safe_metadata(self):
        def entry(version, architecture="amd64", filename="./driver.deb"):
            return (f"Package: libnvidia-gl-580\nArchitecture: {architecture}\nVersion: {version}-1ubuntu1\n"
                    f"Filename: {filename}\nSHA256: " + "a" * 64 + "\n\n")
        wrong = entry("580.126.20", "i386") + entry("580.126.21")
        self.assertIsNone(runtime.driver_package("580.126.20", wrong))
        self.assertEqual(runtime.driver_package("580.126.20", wrong + entry("580.126.20")),
                         (runtime.NVIDIA_REPO + "driver.deb", "a" * 64))
        self.assertEqual(runtime.driver_package("580.126.20", entry("580.126.20").replace("\n", "\r\n")),
                         (runtime.NVIDIA_REPO + "driver.deb", "a" * 64))
        with self.assertRaisesRegex(RuntimeError, "invalid filename"):
            runtime.driver_package("580.126.20", entry("580.126.20", filename="../../driver.deb"))

    def test_native_setup_runs_with_dlss_disabled_and_keeps_its_own_error(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"H3_ENABLE_DLSSNR": "0"}, clear=True):
            root = Path(directory)
            runtime.write_manifest(root, {"error": "DLSS failed independently"})
            with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(runtime.platform, "machine", return_value="x86_64"), patch.object(runtime, "NGX_DIRS", ()), patch.object(runtime, "host_driver_version", return_value="580.126.20"), patch.object(runtime, "driver_components", return_value=root / "drivers/580.126.20"), patch.object(runtime, "installation_lock", side_effect=lambda root: nullcontext()):
                runtime.prepare_runtime(root)
            manifest = runtime.read_manifest(root)
            self.assertEqual(manifest["error"], "DLSS failed independently")
            self.assertEqual(manifest["vsr"]["driver_version"], "580.126.20")
            env = runtime.native_env(root, {"LD_LIBRARY_PATH": "/existing/libs", "KEEP": "value"})
            self.assertEqual(env["KEEP"], "value")
            self.assertTrue(env["LD_LIBRARY_PATH"].endswith(":/existing/libs"))
            self.assertEqual(env["H3_VSR_NGX_LIBRARY"], manifest["vsr"]["library"])

    def test_native_failure_clears_stale_library_without_overwriting_dlss_state(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            runtime.write_manifest(root, {"vsr": {"library": "/old/ngx"}, "initialized": True})
            with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(runtime.platform, "machine", return_value="x86_64"), patch.object(runtime, "host_driver_version", side_effect=RuntimeError("host unavailable")), patch.object(runtime, "installation_lock", side_effect=lambda root: nullcontext()):
                runtime.prepare_native_runtime(root)
            manifest = runtime.read_manifest(root)
            self.assertEqual(manifest["vsr"], {"error": "host unavailable"})
            self.assertTrue(manifest["initialized"])

    def test_dlss_also_prepares_native_ngx_when_vsr_is_disabled(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"H3_ENABLE_NVIDIA_VSR": "0"}, clear=True):
            root = Path(directory)
            with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(runtime.platform, "machine", return_value="x86_64"), patch.object(runtime, "NGX_DIRS", ()), patch.object(runtime, "host_driver_version", return_value="580.126.20"), patch.object(runtime, "driver_components", return_value=root / "drivers/580.126.20") as prepare, patch.object(runtime, "installation_lock", side_effect=lambda root: nullcontext()):
                runtime.prepare_native_runtime(root)
            prepare.assert_called_once_with(root, "580.126.20")
            self.assertIn("library", runtime.read_manifest(root)["vsr"])

    def test_http_404_includes_url_and_does_not_retry_missing_artifact(self):
        url = "https://example.invalid/missing.deb"
        with TemporaryDirectory() as directory, patch.object(runtime, "urlopen", side_effect=runtime.HTTPError(url, 404, "Not Found", {}, None)) as fetch:
            with self.assertRaisesRegex(RuntimeError, "https://example.invalid/missing.deb.*404"):
                runtime.download(url, Path(directory) / "missing.deb")
            fetch.assert_called_once()

    def test_colab_glibc_uses_compatible_wine_and_newer_hosts_reuse_proton(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proton = root / "proton"
            with patch.object(runtime.platform, "libc_ver", return_value=("glibc", "2.35")), patch.object(runtime, "install_archive", return_value=root / "compatible") as install:
                wine, version = runtime.choose_wine_runner(root, proton)
                self.assertEqual(wine, root / "compatible/bin/wine")
                self.assertEqual(version, runtime.WINE_COMPAT_VERSION)
                self.assertIn("-amd64-wow64", install.call_args.args[1])
                self.assertEqual(install.call_args.args[3], runtime.WINE_COMPAT_SHA256)
            with patch.object(runtime.platform, "libc_ver", return_value=("glibc", "2.39")), patch.object(runtime, "install_archive") as install:
                self.assertEqual(runtime.choose_wine_runner(root, proton)[0], proton / "files/bin/wine")
                install.assert_not_called()

    def test_headless_launch_wraps_worker_and_keeps_arguments_separate(self):
        command = ["/runner/wine", "/my helper/video2dlssnr.exe", "--nr-video"]
        with patch.object(runtime.os, "name", "posix"), patch.dict(os.environ, {}, clear=True), patch.object(runtime.shutil, "which", return_value="/usr/bin/xvfb-run"):
            wrapped = runtime.headless_command(command)
        self.assertEqual(wrapped[-3:], command)
        self.assertEqual(wrapped[:2], ["/usr/bin/xvfb-run", "-a"])
        with patch.object(runtime.os, "name", "posix"), patch.dict(os.environ, {"DISPLAY": ":99"}, clear=True):
            self.assertEqual(runtime.headless_command(command), command)

    def test_prefix_installs_native_graphics_and_matching_bridges_then_stops_server(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prefix, proton, bridges = root / "prefix", root / "proton", root / "bridges"
            prefix.mkdir()
            (prefix / "system.reg").touch()
            kernel = prefix / "drive_c/windows/system32/kernel32.dll"
            kernel.parent.mkdir(parents=True)
            kernel.touch()
            bridges.mkdir()
            for name in ("nvngx.dll", "_nvngx.dll"):
                (bridges / name).write_bytes(name.encode())
            for name, package in runtime.TRANSLATION_DLLS.items():
                path = proton / "files/lib/wine" / package / "x86_64-windows" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"native " + name.encode())
            env = {"H3_DLSSNR_WINEPREFIX": str(prefix), "H3_DLSSNR_WINE": str(proton / "files/bin/wine")}
            with patch.object(runtime, "driver_bridges", return_value=(bridges, "610.50")), patch.object(runtime, "wine_run") as run, patch.object(runtime.subprocess, "run", return_value=Mock(returncode=0, stderr="", stdout="")) as stop:
                version = runtime.initialize_prefix(root, proton, env)
            self.assertEqual(version, "610.50")
            self.assertEqual(run.call_count, 4)
            self.assertTrue(all(call.args[1][0] == r"C:\windows\system32\reg.exe" for call in run.call_args_list))
            self.assertEqual((prefix / "drive_c/windows/system32/dxgi.dll").read_bytes(), b"native dxgi.dll")
            self.assertTrue((prefix / "drive_c/ngx_bridge/_nvngx.dll").is_file())
            self.assertEqual(stop.call_args.args[0][-1], "-k")
            self.assertEqual(stop.call_args.kwargs["env"]["WINEPREFIX"], str(prefix))
            self.assertIn("nvapi64", stop.call_args.kwargs["env"]["WINEDLLOVERRIDES"])

    def test_proton_template_repairs_partial_prefix_without_overwriting_registry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proton, prefix = root / "proton", root / "prefix"
            template = proton / "files/share/default_pfx"
            kernel = Path("drive_c/windows/system32/kernel32.dll")
            (template / kernel).parent.mkdir(parents=True)
            (template / kernel).write_bytes(b"builtin")
            (template / "system.reg").write_text("template registry")
            inf = proton / "files/share/wine/wine.inf"
            inf.parent.mkdir(parents=True)
            inf.touch()
            prefix.mkdir()
            (prefix / "system.reg").write_text("existing registry")
            # Windows cannot create Wine's c: and z: device names.
            with patch.object(Path, "symlink_to"):
                runtime.seed_proton_prefix(proton, prefix)
            self.assertEqual((prefix / kernel).read_bytes(), b"builtin")
            self.assertEqual((prefix / "system.reg").read_text(), "existing registry")
            self.assertEqual((prefix / ".update-timestamp").read_text().strip(), str(int(inf.stat().st_mtime)))
            env = {"H3_DLSSNR_WINEPREFIX": str(prefix), "H3_DLSSNR_WINE": str(proton / "files/bin/wine")}
            (prefix / kernel).unlink()
            with patch.object(runtime, "seed_proton_prefix") as seed, patch.object(runtime, "driver_bridges", side_effect=RuntimeError("driver unavailable")), patch.object(runtime, "copy_dll"), patch.object(runtime, "wine_run") as boot, patch.object(runtime, "stop_wine_server"):
                with self.assertRaisesRegex(RuntimeError, "driver unavailable"):
                    runtime.initialize_prefix(root, proton, env)
            seed.assert_called_once_with(proton, prefix)
            boot.assert_not_called()

    def test_prefix_failure_still_stops_only_its_server(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"H3_DLSSNR_WINEPREFIX": str(root / "prefix"), "H3_DLSSNR_WINE": str(root / "runner/bin/wine")}
            with patch.object(runtime, "wine_run", side_effect=RuntimeError("boot failed")), patch.object(runtime.subprocess, "run", return_value=Mock(returncode=1, stderr="", stdout="")) as stop:
                with self.assertRaisesRegex(RuntimeError, "boot failed"):
                    runtime.initialize_prefix(root, root / "runner", env)
            stop.assert_called_once()
            self.assertEqual(stop.call_args.args[0], [str(root / "runner/bin/wineserver"), "-k"])

    def test_cleanup_accepts_already_exited_server_and_reports_real_failures(self):
        with patch.object(runtime.subprocess, "run", return_value=Mock(returncode=1, stderr="", stdout="")):
            runtime.stop_wine_server("/runner/bin/wine", {"WINEPREFIX": "/dedicated"})
        with patch.object(runtime.subprocess, "run", return_value=Mock(returncode=1, stderr="could not load library", stdout="")):
            with self.assertRaisesRegex(RuntimeError, "could not load library"):
                runtime.stop_wine_server("/runner/bin/wine", {"WINEPREFIX": "/dedicated"})

    def test_cleanup_failure_never_masks_initialization_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"H3_DLSSNR_WINEPREFIX": str(root / "prefix"), "H3_DLSSNR_WINE": str(root / "bin/wine")}
            with patch.object(runtime, "wine_run", side_effect=RuntimeError("original Wine loader error")), patch.object(runtime, "stop_wine_server", side_effect=RuntimeError("secondary cleanup error")):
                with self.assertRaisesRegex(RuntimeError, "original Wine loader error"):
                    runtime.initialize_prefix(root, root, env)

    def test_live_host_initialization_is_separate_from_image_build(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            concrete_path = type(root)
            helper, proton = root / "helper", root / "proton"
            helper.mkdir()
            (helper / "nvngx_dlssnr.dll").write_bytes(b"model")
            stack.enter_context(patch.object(runtime, "Path", concrete_path))
            stack.enter_context(patch.object(runtime.os, "name", "posix"))
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(patch.object(runtime.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(runtime.platform, "machine", return_value="x86_64"))
            stack.enter_context(patch.object(runtime, "installation_lock", side_effect=lambda root: nullcontext()))
            stack.enter_context(patch.object(runtime, "install_archive", side_effect=[helper, proton, helper, proton]))
            packages = stack.enter_context(patch.object(runtime, "install_linux_packages"))
            stack.enter_context(patch.object(runtime, "choose_wine_runner", return_value=(proton / "files/bin/wine", "GE")))
            initialize = stack.enter_context(patch.object(runtime, "initialize_prefix", return_value="610.50"))
            runtime.ensure_runtime(root, system_packages=False)
            initialize.assert_not_called()
            self.assertFalse(runtime.read_manifest(root)["initialized"])
            runtime.write_manifest(root, dict(runtime.read_manifest(root), error="previous setup failure"))
            runtime.prepare_runtime(root)
            initialize.assert_called_once()
            packages.assert_called_once()
            self.assertTrue(runtime.read_manifest(root)["initialized"])
            self.assertNotIn("error", runtime.read_manifest(root))


class LauncherProvisioningTests(unittest.TestCase):
    def test_colab_launches_shared_installer_and_modal_packages_same_inputs(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "minimax_h3_colab.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("exec env -u PYTHONPATH", source)
        self.assertIn("run_h3.sh", source)
        launcher = (root / "run_h3.sh").read_text(encoding="utf-8")
        self.assertIn('"$SCRIPT_DIR/h3_nvidia_runtime.py" --initialize', launcher)
        self.assertLess(launcher.index('"$SCRIPT_DIR/h3_nvidia_runtime.py" --initialize'), launcher.index('log "Starting ComfyUI'))
        modal = (root / "modal_h3.py").read_text(encoding="utf-8")
        tree = ast.parse(modal)
        build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build")
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "ensure_runtime" for node in ast.walk(build)))
        self.assertIn("(LOCAL_NVIDIA_RUNTIME, NVIDIA_RUNTIME)", modal)
        self.assertIn("prepare_nvidia_runtime()", modal)


if __name__ == "__main__":
    unittest.main()
