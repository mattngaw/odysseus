"""Deployment transactions against temporary files and fake services; no host changes."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

CONTROL = Path(__file__).with_name("periplousctl")
loader = importlib.machinery.SourceFileLoader("periplous_control", str(CONTROL))
spec = importlib.util.spec_from_loader(loader.name, loader)
ctl = importlib.util.module_from_spec(spec)
loader.exec_module(ctl)


class Releases(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="periplous-release-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        (self.source / "crates/periplous").mkdir(parents=True)
        for path in ("Cargo.toml", "Cargo.lock", "crates/periplous/source.rs"):
            (self.source / path).write_text("fixture")
        self.provenance = self.root / "provenance.json"
        ctl.atomic_json(self.provenance, ctl.provenance(self.source))
        self.calls = []
        self.running = {"dev": False, "prod": False}
        self.bad = set()
        self.manager = ctl.Manager(self.root / "installed", self.service)
        self.manager.ready = self.ready
        for environment, port in (("dev", 8766), ("prod", 8765)):
            ctl.atomic_json(self.manager.path(environment, "config.json"), {"bind": "127.0.0.1", "port": port})
        self.a = self.make_release(b"first binary")
        self.b = self.make_release(b"second binary")

    def make_release(self, data):
        binary = self.root / "binary"
        binary.write_bytes(data)
        package = ctl.package(binary, self.source, self.provenance, self.root / "packages")
        return self.manager.stage(package)

    def service(self, *args, **kwargs):
        self.calls.append(args)
        environment = next((env for env in ctl.ENVIRONMENTS if ctl.unit(env) in args), None)
        if args[0] in ("restart", "start"):
            self.running[environment] = True
        if args[0] == "stop":
            self.running[environment] = False
        code = int(not self.running[environment]) if args[0] == "is-active" else 0
        return subprocess.CompletedProcess(args, code)

    def ready(self, environment, release):
        self.manager.release(release)
        if (environment, release) in self.bad or not self.running[environment]:
            raise RuntimeError("unhealthy test service")
        return "http://test"

    def initial_prod(self):
        self.manager.bootstrap("prod", self.a)
        self.running["prod"] = True

    def test_content_addressing_and_idempotent_stage(self):
        self.assertEqual(self.a, self.make_release(b"first binary"))
        self.assertNotEqual(self.a, self.b)
        manifest = ctl.verify(self.manager.release(self.a), self.a)
        self.assertIsNone(manifest["git_commit"])
        self.assertIsNone(manifest["git_dirty"])

    def test_first_start_tolerates_an_unloaded_systemd_template_instance(self):
        original = self.manager.control
        def control(*args, **kwargs):
            if args[0] == "reset-failed":
                if kwargs.get("check", True):
                    raise subprocess.CalledProcessError(1, args)
                return subprocess.CompletedProcess(args, 1)
            return original(*args, **kwargs)
        self.manager.control = control
        self.manager.switch("dev", self.a)
        self.assertEqual(self.manager.state("dev")["current"], self.a)

    def test_tampering_and_path_traversal_are_rejected(self):
        for bad in ("../../elsewhere", "r-abc", "", None):
            with self.assertRaises(ValueError):
                self.manager.release(bad)
        binary = self.manager.release(self.a) / "periplous"
        binary.chmod(0o600)
        binary.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "binary checksum"):
            self.manager.switch("dev", self.a)
        self.assertEqual(self.manager.state("dev")["current"], None)
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):
            self.manager.stage(self.root / "packages" / self.a)

    def test_modified_sources_cannot_reuse_provenance(self):
        (self.source / "crates/periplous/source.rs").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.make_release(b"first binary")

    def test_dev_deploy_is_isolated_and_promotion_uses_exact_artifact(self):
        self.initial_prod()
        self.calls.clear()
        self.manager.switch("dev", self.b)
        self.assertEqual(self.manager.state("prod")["current"], self.a)
        self.assertFalse(any(ctl.unit("prod") in call for call in self.calls))
        self.manager.promote()
        self.assertEqual(self.manager.state("prod"), {"current": self.b, "previous": self.a})
        self.assertEqual(self.manager.state("dev")["current"], self.b)
        self.manager.rollback("prod")
        self.assertEqual(self.manager.state("prod"), {"current": self.a, "previous": self.b})
        self.manager.promote()
        self.assertEqual(self.manager.state("prod")["current"], self.b)
        self.assertFalse(any(ctl.TUNNEL in call for call in self.calls))

    def test_unhealthy_dev_cannot_be_promoted(self):
        self.initial_prod()
        self.manager.bootstrap("dev", self.b)
        with self.assertRaises(RuntimeError):
            self.manager.promote()
        self.assertEqual(self.manager.state("prod")["current"], self.a)

    def test_failed_promotion_restores_old_service_and_history(self):
        self.initial_prod()
        self.manager.switch("dev", self.b)
        self.bad.add(("prod", self.b))
        with self.assertRaises(RuntimeError):
            self.manager.promote()
        self.assertEqual(self.manager.state("prod"), {"current": self.a, "previous": None})
        self.assertTrue(self.running["prod"])
        self.assertEqual(self.calls[-1], ("restart", ctl.unit("prod")))

    def test_failed_first_deploy_returns_to_empty_stopped_state(self):
        self.bad.add(("dev", self.a))
        with self.assertRaises(RuntimeError):
            self.manager.switch("dev", self.a)
        self.assertEqual(self.manager.state("dev")["current"], None)
        self.assertFalse(self.running["dev"])

    def test_interrupted_switch_requires_recovery_and_recovery_is_retryable(self):
        self.initial_prod()
        old = self.manager.state("prod")
        ctl.atomic_json(self.manager.path("prod", "state.json"), {
            "current": self.b, "previous": None,
            "pending": {"state": old, "was_active": True}})
        with self.assertRaisesRegex(RuntimeError, "recover"):
            self.manager.switch("prod", self.a)
        self.bad.add(("prod", self.a))
        with self.assertRaises(RuntimeError):
            self.manager.recover("prod")
        self.assertIn("pending", self.manager.state("prod"))
        self.bad.clear()
        self.manager.recover("prod")
        self.assertEqual(self.manager.state("prod"), old)

    def test_system_state_is_readable_but_release_files_are_not_writable(self):
        system = ctl.Manager(self.root / "system", control=lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
                             unit_name=lambda env: "periplous-prod.service", public_state=True)
        system.stage(self.root / "packages" / self.a)
        system.bootstrap("prod", self.a)
        self.assertEqual(system.path("prod", "state.json").stat().st_mode & 0o777, 0o644)
        self.assertEqual(system.release(self.a).stat().st_mode & 0o777, 0o755)
        self.assertEqual((system.release(self.a) / "periplous").stat().st_mode & 0o777, 0o555)

    def test_launch_for_socket_activation_preserves_activation_environment(self):
        self.manager.bootstrap("prod", self.a)
        ctl.atomic_json(self.manager.path("prod", "config.json"), {
            "bind": "0.0.0.0", "port": 8765, "socket_activation": True,
            "allowed_hosts": ["localhost", "198.51.100.32"]})
        with patch.dict(ctl.os.environ, {"LISTEN_PID": "123", "LISTEN_FDS": "1", "LISTEN_FDNAMES": "http"}), patch.object(ctl.os, "execve") as execute:
            self.manager.launch("prod")
        binary, args, environment = execute.call_args.args
        self.assertEqual(args, [str(binary), "serve", "--systemd"])
        self.assertEqual(environment["LISTEN_FDS"], "1")
        self.assertEqual(environment["PERIPLOUS_ALLOWED_HOSTS"], "localhost,198.51.100.32")
        self.assertEqual(environment["PERIPLOUS_ENVIRONMENT"], "prod")

    def test_rollback_and_bootstrap_guardrails(self):
        with self.assertRaises(ValueError):
            self.manager.rollback("prod")
        self.initial_prod()
        with self.assertRaises(ValueError):
            self.manager.bootstrap("prod", self.b)


class CommandBoundary(unittest.TestCase):
    def test_invalid_arguments_do_not_reach_service_control(self):
        for args in [[], ["tunnel start"], ["start", "extra"], ["deploy", "prod", "r-id"],
                     ["tunnel"], ["tunnel", "enable"], ["destroy"]]:
            with self.subTest(args=args), patch("sys.argv", [str(CONTROL), *args]), \
                    patch.object(ctl, "service") as service, patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    ctl.main()
                self.assertEqual(error.exception.code, 2)
                service.assert_not_called()

    def test_url_uses_only_active_tunnel_invocation(self):
        def service(*args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="current-invocation\n")
        with patch.object(ctl, "service", service), patch.object(ctl.subprocess, "check_output",
                return_value="https://current.trycloudflare.com") as journal:
            self.assertEqual(ctl.tunnel_url(), "https://current.trycloudflare.com")
            self.assertIn("_SYSTEMD_INVOCATION_ID=current-invocation", journal.call_args.args[0])
        with patch.object(ctl, "service", return_value=subprocess.CompletedProcess([], 3)), \
                patch.object(ctl.subprocess, "check_output") as journal:
            with self.assertRaises(RuntimeError):
                ctl.tunnel_url()
            journal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
