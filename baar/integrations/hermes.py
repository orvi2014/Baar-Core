"""
baar/integrations/hermes.py — Hermes Agent integration.

Wires baar-core as a local OpenAI-compatible proxy that Hermes Agent routes
every LLM call through.  Hermes handles agentic work (tool use, planning,
multi-step reasoning); baar enforces a hard budget cap pre-flight on every turn.

Architecture:

    BaarHermesSession(budget=1.00).__enter__()
      ├─ BaarHermesProxy.start()    # uvicorn on 127.0.0.1:8080, daemon thread
      └─ launch `hermes` subprocess # HERMES_HOME → temp dir pointing to proxy

    Each Hermes LLM turn:
      POST http://127.0.0.1:8080/v1/chat/completions
        └─ BAARRouter.achat()
              ├─ pre-flight budget check          (BudgetExhausted → 402)
              ├─ complexity routing → small/big
              └─ real provider call via LiteLLM

Install:
    pip install baar-core[vercel] hermes-agent
    # hermes-agent installs the `hermes` CLI

Quick start:
    from baar.integrations.hermes import BaarHermesSession

    with BaarHermesSession(budget=1.00) as session:
        session.run_interactive()       # interactive hermes chat, budget-capped
        # or:
        reply = session.run_task("Research the top 3 open-source agent frameworks")
        print(reply)
        print(f"Spent ${session.spent:.4f} of $1.00")

Permanent setup (run once, then just use `hermes` normally):
    from baar.integrations.hermes import setup_baar_provider, BaarHermesProxy
    from baar import BAARRouter

    router = BAARRouter(budget=5.00)
    proxy = BaarHermesProxy(router, port=8080)
    proxy.start()
    setup_baar_provider(port=8080)
    # Now: hermes  ← all LLM calls go through baar
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from baar.router import BAARRouter
from baar.core.models import RoutingLog

try:
    from baar.integrations.vercel import create_app
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI and uvicorn are required for the Hermes integration. "
        "Install with:  pip install baar-core[vercel]"
    ) from exc


# ── BaarHermesProxy ───────────────────────────────────────────────────────────

class BaarHermesProxy:
    """
    Runs baar-core as an OpenAI-compatible HTTP proxy in a background thread.

    Any OpenAI-compatible client (including Hermes Agent) can point its
    base_url at this proxy and get baar's routing + hard budget enforcement
    transparently.

    Args:
        router:  A configured BAARRouter instance.
        port:    Local port for the proxy server (default: 8080).
    """

    def __init__(self, router: BAARRouter, port: int = 8080) -> None:
        self._router = router
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._server = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    def start(self, timeout: float = 10.0) -> None:
        """Start the proxy server and block until it accepts connections."""
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError(
                "uvicorn is required. Install with:  pip install baar-core[vercel]"
            ) from exc

        app = create_app(self._router)
        ready = threading.Event()

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self._port,
                log_level="warning",
                loop="asyncio",
            )
            self._server = uvicorn.Server(config)

            async def _serve():
                await self._server.startup()
                ready.set()
                await self._server.main_loop()
                await self._server.shutdown()

            loop.run_until_complete(_serve())

        self._thread = threading.Thread(target=_run, daemon=True, name="baar-hermes-proxy")
        self._thread.start()

        # Wait until the port is actually accepting connections
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=0.2):
                    return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.1)

        raise RuntimeError(
            f"Baar proxy server did not start on port {self._port} within {timeout}s"
        )

    def stop(self) -> None:
        """Signal the server to stop and wait for the thread to exit."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


# ── BaarHermesSession ─────────────────────────────────────────────────────────

class BaarHermesSession:
    """
    Context manager that starts a budget-capped Hermes Agent session.

    On entry:
      1. Starts BaarHermesProxy in a daemon thread.
      2. Writes a temporary HERMES_HOME with a config.yaml that points
         Hermes to the baar proxy — the user's real ~/.hermes is untouched.

    On exit:
      - Terminates any running Hermes subprocess.
      - Stops the proxy server.
      - Cleans up the temporary HERMES_HOME.

    Args:
        budget:       Hard spending cap in USD.
        port:         Local port for the proxy (default: 8080).
        small_model:  Cheap model for simple tasks (default: "gpt-4o-mini").
        big_model:    Capable model for complex tasks (default: "gpt-4o").
        hermes_home:  Optional explicit path for the temp HERMES_HOME directory.
                      Created automatically if None; deleted on exit.

    Example:
        with BaarHermesSession(budget=1.00) as session:
            reply = session.run_task("Summarise the top 5 AI papers this week")
            print(reply)
            print(f"Spent ${session.spent:.4f}")
    """

    def __init__(
        self,
        budget: float,
        port: int = 8080,
        small_model: str = "gpt-4o-mini",
        big_model: str = "gpt-4o",
        hermes_home: Optional[str] = None,
    ) -> None:
        self._router = BAARRouter(
            budget=budget,
            small_model=small_model,
            big_model=big_model,
        )
        self._proxy = BaarHermesProxy(self._router, port=port)
        self._hermes_home = hermes_home
        self._owns_hermes_home = hermes_home is None
        self._hermes_proc: Optional[subprocess.Popen] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "BaarHermesSession":
        if self._owns_hermes_home:
            self._hermes_home = tempfile.mkdtemp(prefix="baar-hermes-")
        self._write_hermes_config()
        self._proxy.start()
        return self

    def __exit__(self, *args) -> None:
        self._terminate_hermes()
        self._proxy.stop()
        if self._owns_hermes_home and self._hermes_home and os.path.isdir(self._hermes_home):
            shutil.rmtree(self._hermes_home, ignore_errors=True)

    # ── Public methods ────────────────────────────────────────────────────────

    def run_interactive(self) -> None:
        """
        Launch Hermes in interactive mode (foreground).
        Blocks until the user exits the Hermes session or budget is exhausted.
        """
        self._hermes_proc = subprocess.Popen(
            ["hermes"],
            env=self._hermes_env(),
        )
        self._hermes_proc.wait()
        self._hermes_proc = None

    def run_task(self, task: str, timeout: int = 300) -> str:
        """
        Run a single task through Hermes Agent non-interactively.

        Passes the task via stdin and returns Hermes's stdout output.
        Raises subprocess.TimeoutExpired if Hermes exceeds timeout seconds.

        Args:
            task:    The task description to send to Hermes.
            timeout: Maximum seconds to wait for completion (default: 300).

        Returns:
            The text output produced by Hermes.
        """
        self._hermes_proc = subprocess.Popen(
            ["hermes"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._hermes_env(),
        )
        try:
            stdout, _ = self._hermes_proc.communicate(
                input=task + "\n/exit\n",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            self._hermes_proc.kill()
            self._hermes_proc.communicate()
            raise
        finally:
            self._hermes_proc = None
        return stdout.strip()

    # ── Budget properties ─────────────────────────────────────────────────────

    @property
    def spent(self) -> float:
        """Total USD spent so far in this session."""
        return self._router.spent

    @property
    def remaining(self) -> float:
        """USD remaining before the kill-switch fires."""
        return self._router.remaining

    @property
    def budget(self) -> float:
        return self._router.budget

    @property
    def log(self) -> RoutingLog:
        """Full routing audit trail for this session."""
        return self._router.log

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _hermes_env(self) -> dict:
        return {**os.environ, "HERMES_HOME": self._hermes_home}

    def _write_hermes_config(self) -> None:
        """Write a minimal Hermes config.yaml pointing at the baar proxy."""
        config = {
            "model": {
                "provider": "openai",
                "name": "baar",
                "base_url": self._proxy.base_url,
                "api_key": "baar-proxy",
            }
        }
        config_path = os.path.join(self._hermes_home, "config.yaml")
        if yaml is not None:
            with open(config_path, "w") as fh:
                yaml.dump(config, fh, default_flow_style=False)
        else:
            # Fallback: write YAML manually (only safe for this simple structure)
            with open(config_path, "w") as fh:
                fh.write("model:\n")
                fh.write("  provider: openai\n")
                fh.write("  name: baar\n")
                fh.write(f"  base_url: \"{self._proxy.base_url}\"\n")
                fh.write("  api_key: baar-proxy\n")

    def _terminate_hermes(self) -> None:
        if self._hermes_proc is not None:
            try:
                self._hermes_proc.terminate()
                self._hermes_proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                self._hermes_proc.kill()
            finally:
                self._hermes_proc = None


# ── setup_baar_provider ───────────────────────────────────────────────────────

def setup_baar_provider(
    port: int = 8080,
    hermes_home: Optional[str] = None,
) -> str:
    """
    Write a Hermes config that permanently points to a running baar proxy.

    Call this once after starting BaarHermesProxy.start().  Subsequent
    invocations of `hermes` (without a BaarHermesSession) will route through
    the proxy at the given port.

    Args:
        port:        Port where BaarHermesProxy is listening (default: 8080).
        hermes_home: Path to write config into (default: ~/.hermes).

    Returns:
        Path to the config.yaml that was written.

    Example:
        router = BAARRouter(budget=5.00)
        proxy = BaarHermesProxy(router, port=8080)
        proxy.start()
        setup_baar_provider(port=8080)
        # Now run: hermes   ← all LLM calls go through baar
    """
    target_dir = hermes_home or os.path.expanduser("~/.hermes")
    os.makedirs(target_dir, exist_ok=True)
    config_path = os.path.join(target_dir, "config.yaml")

    base_url = f"http://127.0.0.1:{port}/v1"

    if yaml is not None:
        existing: dict = {}
        if os.path.exists(config_path):
            with open(config_path) as fh:
                existing = yaml.safe_load(fh) or {}
        existing["model"] = {
            "provider": "openai",
            "name": "baar",
            "base_url": base_url,
            "api_key": "baar-proxy",
        }
        with open(config_path, "w") as fh:
            yaml.dump(existing, fh, default_flow_style=False)
    else:
        # Fallback: overwrite only the model block
        with open(config_path, "w") as fh:
            fh.write("model:\n")
            fh.write("  provider: openai\n")
            fh.write("  name: baar\n")
            fh.write(f"  base_url: \"{base_url}\"\n")
            fh.write("  api_key: baar-proxy\n")

    return config_path
