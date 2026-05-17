"""
tests/test_hermes_integration.py — Unit tests for baar/integrations/hermes.py

All tests mock the Hermes subprocess and uvicorn server so they run without
a real Hermes installation or live LLM provider.
"""

import asyncio
import os
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from baar import BAARRouter
from baar.integrations.hermes import BaarHermesProxy, BaarHermesSession, setup_baar_provider
from baar.core.exceptions import BudgetExhausted


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return BAARRouter(budget=1.00, small_model="gpt-4o-mini", big_model="gpt-4o")


@pytest.fixture
def free_port():
    """Return an available localhost port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── BaarHermesProxy tests ─────────────────────────────────────────────────────

class TestBaarHermesProxy:
    def test_base_url_format(self, router, free_port):
        proxy = BaarHermesProxy(router, port=free_port)
        assert proxy.base_url == f"http://127.0.0.1:{free_port}/v1"

    def test_start_times_out_on_dead_port(self, router):
        proxy = BaarHermesProxy(router, port=19999)

        with patch("baar.integrations.hermes.socket.create_connection") as mock_conn:
            mock_conn.side_effect = ConnectionRefusedError
            with patch("baar.integrations.hermes.threading.Thread"):
                with pytest.raises(RuntimeError, match="did not start"):
                    proxy.start(timeout=0.3)

    def test_stop_sets_server_should_exit(self, router, free_port):
        proxy = BaarHermesProxy(router, port=free_port)
        mock_server = MagicMock()
        proxy._server = mock_server
        mock_thread = MagicMock()
        proxy._thread = mock_thread

        proxy.stop()

        assert mock_server.should_exit is True
        mock_thread.join.assert_called_once_with(timeout=5)
        assert proxy._thread is None

    def test_stop_is_noop_when_never_started(self, router, free_port):
        proxy = BaarHermesProxy(router, port=free_port)
        proxy.stop()  # should not raise

    def test_start_spawns_daemon_thread(self, router, free_port):
        proxy = BaarHermesProxy(router, port=free_port)

        with patch("baar.integrations.hermes.threading.Thread") as MockThread:
            mock_t = MagicMock()
            MockThread.return_value = mock_t

            with patch("baar.integrations.hermes.socket.create_connection"):
                proxy.start(timeout=1.0)

            MockThread.assert_called_once()
            _, kwargs = MockThread.call_args
            assert kwargs.get("daemon") is True
            mock_t.start.assert_called_once()

    def test_start_raises_import_error_when_uvicorn_missing(self, router, free_port):
        proxy = BaarHermesProxy(router, port=free_port)

        import builtins
        real_import = builtins.__import__

        def _block_uvicorn(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("No module named 'uvicorn'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_uvicorn):
            with pytest.raises(ImportError, match="uvicorn is required"):
                proxy.start()

    def test_run_thread_body_calls_uvicorn_lifecycle(self, router, free_port):
        """The _run() target function sets up and drives the uvicorn server."""
        proxy = BaarHermesProxy(router, port=free_port)

        captured_target = []

        def _capture_thread(*args, **kwargs):
            t = MagicMock()
            t._target = kwargs.get("target")
            captured_target.append(kwargs.get("target"))
            return t

        with patch("baar.integrations.hermes.threading.Thread", side_effect=_capture_thread):
            with patch("baar.integrations.hermes.socket.create_connection"):
                proxy.start(timeout=1.0)

        assert captured_target, "Thread target was not captured"
        run_fn = captured_target[0]

        mock_server = MagicMock()
        mock_server.startup = AsyncMock()
        mock_server.main_loop = AsyncMock()
        mock_server.shutdown = AsyncMock()

        with patch("uvicorn.Config"), \
             patch("uvicorn.Server", return_value=mock_server):
            run_fn()

        mock_server.startup.assert_awaited_once()
        mock_server.main_loop.assert_awaited_once()
        mock_server.shutdown.assert_awaited_once()


# ── BaarHermesSession tests ───────────────────────────────────────────────────

class TestBaarHermesSession:
    def _make_session(self, budget=1.00, port=8081, tmpdir=None):
        return BaarHermesSession(budget=budget, port=port, hermes_home=tmpdir)

    @pytest.fixture
    def tmp_hermes_home(self, tmp_path):
        return str(tmp_path / "hermes-home")

    def test_context_manager_creates_and_removes_tempdir(self, free_port):
        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port) as session:
                    assert session._hermes_home is not None
                    assert os.path.isdir(session._hermes_home)
                    tmp = session._hermes_home
                assert not os.path.exists(tmp)

    def test_context_manager_does_not_remove_user_provided_dir(self, free_port, tmp_path):
        user_dir = str(tmp_path / "my-hermes")
        os.makedirs(user_dir)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=user_dir):
                    pass

        assert os.path.isdir(user_dir), "User-provided HERMES_HOME should not be deleted"

    def test_writes_config_yaml(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)
        session = self._make_session(port=free_port, tmpdir=hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with session:
                    config_path = os.path.join(hermes_home, "config.yaml")
                    assert os.path.exists(config_path)
                    content = Path(config_path).read_text()
                    assert "provider" in content
                    assert f"127.0.0.1:{free_port}" in content
                    assert "baar-proxy" in content

    def test_writes_config_without_yaml_module(self, free_port, tmp_path):
        """When pyyaml is not installed, config is written as plain text."""
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)
        session = self._make_session(port=free_port, tmpdir=hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with patch("baar.integrations.hermes.yaml", None):
                    with session:
                        config_path = os.path.join(hermes_home, "config.yaml")
                        content = Path(config_path).read_text()
                        assert "model:" in content
                        assert "provider: openai" in content
                        assert f"127.0.0.1:{free_port}" in content
                        assert "baar-proxy" in content

    def test_hermes_env_sets_hermes_home(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)
        session = self._make_session(port=free_port, tmpdir=hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with session:
                    env = session._hermes_env()
                    assert env["HERMES_HOME"] == hermes_home

    def test_real_hermes_home_not_touched(self, free_port):
        """HERMES_HOME in subprocess env must be our temp dir, never ~/.hermes."""
        real_home = os.path.expanduser("~/.hermes")

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port) as session:
                    env = session._hermes_env()
                    assert env.get("HERMES_HOME") != real_home

    def test_run_interactive_launches_hermes_and_waits(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=hermes_home) as session:
                    with patch("subprocess.Popen") as MockPopen:
                        mock_proc = MagicMock()
                        mock_proc.wait.return_value = 0
                        MockPopen.return_value = mock_proc

                        session.run_interactive()

                        MockPopen.assert_called_once_with(["hermes"], env=session._hermes_env())
                        mock_proc.wait.assert_called_once()
                        assert session._hermes_proc is None

    def test_run_task_passes_task_via_stdin(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=hermes_home) as session:
                    with patch("subprocess.Popen") as MockPopen:
                        mock_proc = MagicMock()
                        mock_proc.communicate.return_value = ("Agent output here", "")
                        MockPopen.return_value = mock_proc

                        result = session.run_task("Summarise AI news", timeout=60)

                        assert result == "Agent output here"
                        stdin_input = mock_proc.communicate.call_args[1].get("input") or \
                                      mock_proc.communicate.call_args[0][0]
                        assert "Summarise AI news" in stdin_input

    def test_run_task_terminates_and_reraises_on_timeout(self, free_port, tmp_path):
        """kill() is called and TimeoutExpired is re-raised after draining stdout."""
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=hermes_home) as session:
                    with patch("subprocess.Popen") as MockPopen:
                        mock_proc = MagicMock()
                        # First communicate() raises; second (drain after kill) succeeds
                        mock_proc.communicate.side_effect = [
                            subprocess.TimeoutExpired(cmd=["hermes"], timeout=1),
                            ("", ""),
                        ]
                        MockPopen.return_value = mock_proc

                        with pytest.raises(subprocess.TimeoutExpired):
                            session.run_task("slow task", timeout=1)

                        mock_proc.kill.assert_called_once()
                        assert mock_proc.communicate.call_count == 2

    def test_budget_properties(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=0.50, port=free_port, hermes_home=hermes_home) as session:
                    assert session.budget == pytest.approx(0.50)
                    assert session.spent == pytest.approx(0.0)
                    assert session.remaining == pytest.approx(0.50)

    def test_log_property_returns_routing_log(self, free_port, tmp_path):
        from baar.core.models import RoutingLog
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=hermes_home) as session:
                    log = session.log
                    assert isinstance(log, RoutingLog)

    def test_exit_terminates_running_hermes_proc(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        mock_proc = MagicMock()

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=hermes_home) as session:
                    session._hermes_proc = mock_proc

        mock_proc.terminate.assert_called_once()

    def test_terminate_kills_when_terminate_raises_process_lookup_error(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        mock_proc = MagicMock()
        mock_proc.terminate.side_effect = ProcessLookupError("no such process")

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=hermes_home) as session:
                    session._hermes_proc = mock_proc

        mock_proc.kill.assert_called_once()

    def test_terminate_kills_when_wait_times_out(self, free_port, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd=["hermes"], timeout=5)

        with patch.object(BaarHermesProxy, "start"):
            with patch.object(BaarHermesProxy, "stop"):
                with BaarHermesSession(budget=1.00, port=free_port, hermes_home=hermes_home) as session:
                    session._hermes_proc = mock_proc

        mock_proc.kill.assert_called_once()


# ── setup_baar_provider tests ─────────────────────────────────────────────────

class TestSetupBaarProvider:
    def test_writes_config_to_specified_dir(self, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        config_path = setup_baar_provider(port=8080, hermes_home=hermes_home)

        assert os.path.exists(config_path)
        content = Path(config_path).read_text()
        assert "127.0.0.1:8080" in content
        assert "baar-proxy" in content
        assert "openai" in content

    def test_creates_dir_if_missing(self, tmp_path):
        hermes_home = str(tmp_path / "new" / "nested" / "dir")
        assert not os.path.exists(hermes_home)

        setup_baar_provider(port=9090, hermes_home=hermes_home)

        assert os.path.isdir(hermes_home)
        config = os.path.join(hermes_home, "config.yaml")
        assert os.path.exists(config)

    def test_returns_config_path(self, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        result = setup_baar_provider(port=8080, hermes_home=hermes_home)

        assert result == os.path.join(hermes_home, "config.yaml")

    def test_port_reflected_in_config(self, tmp_path):
        hermes_home = str(tmp_path / "hermes")
        setup_baar_provider(port=12345, hermes_home=hermes_home)

        content = Path(os.path.join(hermes_home, "config.yaml")).read_text()
        assert "12345" in content

    def test_preserves_existing_config_keys_when_yaml_available(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml as _yaml

        hermes_home = str(tmp_path / "hermes")
        os.makedirs(hermes_home)

        existing = {"terminal": {"backend": "docker"}}
        config_path = os.path.join(hermes_home, "config.yaml")
        with open(config_path, "w") as fh:
            _yaml.dump(existing, fh)

        setup_baar_provider(port=8080, hermes_home=hermes_home)

        with open(config_path) as fh:
            merged = _yaml.safe_load(fh)

        assert merged["terminal"]["backend"] == "docker", "Existing keys should be preserved"
        assert merged["model"]["provider"] == "openai"

    def test_writes_plain_text_when_yaml_module_absent(self, tmp_path):
        """Fallback path: no pyyaml, write YAML manually."""
        hermes_home = str(tmp_path / "hermes")

        with patch("baar.integrations.hermes.yaml", None):
            config_path = setup_baar_provider(port=7777, hermes_home=hermes_home)

        content = Path(config_path).read_text()
        assert "model:" in content
        assert "provider: openai" in content
        assert "7777" in content
        assert "baar-proxy" in content
