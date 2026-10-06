"""Optional Colab DriveFS mount using user-owned, reusable OAuth credentials.

DriveFS's metadata interface is internal to Colab. No credentials are written
to disk, logs, ordinary notebook outputs, or widget state by this module.
"""

import base64
import hashlib
import http.server
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


SECRET_NAME = "H3_DRIVE_AUTH"
SCOPES = "email https://www.googleapis.com/auth/drive"
REDIRECT_URI = "http://127.0.0.1:8765/"
_mounts = {}


class DriveAuthError(RuntimeError):
    """Safe user-facing error with no credential payload."""


class ReauthorizationRequired(DriveAuthError):
    """Credentials need replacement rather than a network retry."""


def validate_credentials(value):
    try:
        credentials = json.loads(value) if isinstance(value, str) else dict(value)
        required = ("client_id", "client_secret", "refresh_token")
        if any(not isinstance(credentials.get(k), str) or not credentials[k].strip() for k in required):
            raise ValueError()
        return {k: credentials[k] for k in required}
    except (TypeError, ValueError):
        raise ReauthorizationRequired(f"{SECRET_NAME} must contain client_id, client_secret, and refresh_token JSON.") from None


def token_request(fields):
    request = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=urllib.parse.urlencode(fields).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 401):
            raise ReauthorizationRequired("Google rejected the credentials. Rerun the Drive cell to replace H3_DRIVE_AUTH.") from None
        raise DriveAuthError("Google token service failed. Retry later; keep your existing secret.") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise DriveAuthError("Could not reach Google token service. Retry later; keep your existing secret.") from None
    if not isinstance(result.get("access_token"), str) or not result["access_token"]:
        raise DriveAuthError("Google did not return an access token.")
    return result


class TokenProvider:
    def __init__(self, credentials):
        self.credentials = validate_credentials(credentials)
        self.lock = threading.Lock()
        self.token = None
        self.expiry = 0

    def get(self):
        with self.lock:
            if time.monotonic() >= self.expiry:
                result = token_request(dict(self.credentials, grant_type="refresh_token"))
                self.token = result["access_token"]
                self.expiry = time.monotonic() + max(1, int(result.get("expires_in", 3600)) - 120)
            return {"access_token": self.token, "expires_in": max(1, int(self.expiry - time.monotonic())),
                    "token_type": "Bearer", "scope": SCOPES}

    def email(self):
        request = urllib.request.Request("https://www.googleapis.com/oauth2/v2/userinfo",
                                         headers={"Authorization": "Bearer " + self.get()["access_token"]})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                email = json.load(response)["email"]
            if not isinstance(email, str) or not email:
                raise ValueError()
            return email
        except (OSError, ValueError, KeyError):
            raise DriveAuthError("Could not read the Drive account email. Check that the OAuth client grants email and Drive access.") from None


def drive_mounted(mount_dir):
    mount_dir = Path(mount_dir)
    return os.path.ismount(mount_dir) and (mount_dir / "MyDrive").is_dir()


def mount_with_credentials(mount_dir, credentials, timeout=90):
    """Mount without invoking Colab's browser credential-propagation flow."""
    mount_dir = Path(mount_dir).absolute()
    if drive_mounted(mount_dir):
        return
    key = str(mount_dir)
    previous = _mounts.get(key)
    if previous:
        raise DriveAuthError("A reusable Drive mount is already running but inaccessible. Restart the runtime before retrying.")
    binary = Path("/opt/google/drive/drive")
    if not binary.is_file() or not Path("/dev/fuse").exists():
        raise DriveAuthError("Reusable Drive mounting requires a managed Colab runtime with DriveFS and /dev/fuse.")
    if " " in key or mount_dir.is_symlink() or (mount_dir.exists() and any(mount_dir.iterdir())):
        raise DriveAuthError("Drive mount path must be an empty directory without spaces or a symbolic link.")
    provider = TokenProvider(credentials)
    provider.get()  # Validate before starting a server or changing mount paths.
    email = provider.email()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path.endswith("/token"):
                    body = json.dumps(provider.get()).encode()
                elif self.path.endswith("/scopes"):
                    body = SCOPES.encode()
                elif self.path.endswith("/email") or "guest-attributes" in self.path:
                    body = email.encode()
                elif "service-accounts" in self.path:
                    body = b"default/\n"
                else:
                    body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "application/json" if self.path.endswith("/token") else "text/plain")
                self.end_headers()
                self.wfile.write(body)
            except DriveAuthError:
                self.send_error(503, "Token refresh failed")

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    process = None
    try:
        mount_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen([
            str(binary),
            "--features=crash_throttle_percentage:100,fuse_max_background:1000,max_read_qps:1000,max_write_qps:1000,max_operation_batch_size:15,max_parallel_push_task_instances:10,opendir_timeout_ms:120000,virtual_folders_omit_spaces:true",
            f"--metadata_server_auth_uri=http://127.0.0.1:{server.server_port}/computeMetadata/v1",
            f"--preferences=trusted_root_certs_file_path:/opt/google/drive/roots.pem,feature_flag_restart_seconds:129600,mount_point_path:{mount_dir}",
        ], env=dict(os.environ, HOME="/root", FUSE_DEV_NAME="/dev/fuse"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if drive_mounted(mount_dir):
                _mounts[key] = (server, process)
                return
            if process.poll() is not None:
                raise DriveAuthError("DriveFS exited before mounting. The reusable mount interface may be unsupported in this runtime.")
            time.sleep(0.25)
        raise DriveAuthError("DriveFS mount timed out. Restart the runtime before retrying.")
    finally:
        if key not in _mounts:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            server.shutdown()
            server.server_close()


def mount_from_secret(mount_dir):
    if drive_mounted(mount_dir):
        return
    from google.colab import userdata
    try:
        value = userdata.get(SECRET_NAME)
    except userdata.SecretNotFoundError:
        value = None
    except userdata.NotebookAccessError:
        raise DriveAuthError("Enable notebook access for H3_DRIVE_AUTH in Colab's Secrets panel, then rerun this cell.") from None
    if value is None:
        credentials = one_time_setup()
    else:
        try:
            credentials = validate_credentials(value)
            TokenProvider(credentials).get()
        except ReauthorizationRequired:
            credentials = one_time_setup()
    mount_with_credentials(mount_dir, credentials)


def authorization_request(client):
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(dict(
        client_id=client["client_id"], redirect_uri=REDIRECT_URI, response_type="code",
        scope=SCOPES, access_type="offline", prompt="consent", state=state,
        code_challenge=challenge, code_challenge_method="S256",
    ))
    return url, state, verifier


def exchange_callback(client, callback, state, verifier):
    parsed = urllib.parse.urlsplit(callback.strip())
    params = urllib.parse.parse_qs(parsed.query)
    if (parsed.scheme, parsed.netloc, parsed.path) != ("http", "127.0.0.1:8765", "/") or params.get("state") != [state]:
        raise DriveAuthError("Paste the full redirect URL from the authorization tab for this setup attempt.")
    if "error" in params or not params.get("code"):
        raise DriveAuthError("Authorization was declined or no code was returned. Run setup again.")
    result = token_request(dict(client, grant_type="authorization_code", code=params["code"][0],
                                redirect_uri=REDIRECT_URI, code_verifier=verifier))
    return validate_credentials(dict(client, refresh_token=result.get("refresh_token")))


def one_time_setup():
    """Transient browser DOM via eval_js; no display() or credential prints."""
    from google.colab import output, userdata
    try:
        stored = userdata.get(SECRET_NAME)
    except userdata.SecretNotFoundError:
        stored = None
    except userdata.NotebookAccessError:
        raise DriveAuthError("Enable notebook access for H3_DRIVE_AUTH in the Secrets panel, then rerun setup.") from None
    if stored is not None:
        try:
            TokenProvider(validate_credentials(stored)).get()
        except ReauthorizationRequired:
            print("[i] Existing H3_DRIVE_AUTH needs replacement. Starting one-time setup.")
        else:
            print("[OK] H3_DRIVE_AUTH works. Authorization setup skipped.")
            return validate_credentials(stored)
    client_text = output.eval_js("""new Promise(resolve => {
      const panel = document.createElement('div');
      const label = document.createElement('p');
      label.textContent = 'First use: enable Google Drive API, create a Desktop app OAuth client, and download its JSON. For an external app, use Production to avoid seven-day test tokens. Choose the JSON below.';
      const help = document.createElement('a'); help.href = 'https://console.cloud.google.com/apis/credentials';
      help.target = '_blank'; help.rel = 'noopener noreferrer'; help.textContent = 'Open Google Cloud setup';
      const input = document.createElement('input'); input.type = 'file'; input.accept = '.json';
      input.onchange = async () => { const text = await input.files[0].text(); panel.remove(); resolve(text); };
      panel.append(label, help, document.createElement('br'), input); document.body.append(panel);
    })""", timeout_sec=600)
    try:
        installed = json.loads(client_text)["installed"]
        client = {k: installed[k] for k in ("client_id", "client_secret")}
        if not all(isinstance(v, str) and v for v in client.values()):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise DriveAuthError("Choose a Desktop OAuth client JSON file containing an installed configuration.") from None
    url, state, verifier = authorization_request(client)
    callback = output.eval_js("""new Promise(resolve => {
      const panel = document.createElement('div');
      const link = document.createElement('a'); link.textContent = 'Authorize Google Drive';
      link.href = AUTH_URL; link.target = '_blank'; link.rel = 'noopener noreferrer';
      const text = document.createElement('p');
      text.textContent = 'After approval, the browser redirects to 127.0.0.1 and may show a connection error. Copy that entire address-bar URL and paste below. No local server or CLI is required.';
      const input = document.createElement('input'); input.type = 'password'; input.style.width = '90%';
      input.autocomplete = 'off'; input.placeholder = 'Paste complete redirect URL';
      const button = document.createElement('button'); button.textContent = 'Finish authorization';
      button.onclick = () => { const value = input.value; input.value = ''; panel.remove(); resolve(value); };
      panel.append(link, text, input, button); document.body.append(panel);
    })""".replace("AUTH_URL", json.dumps(url)), timeout_sec=600)
    credentials = exchange_callback(client, callback, state, verifier)
    TokenProvider(credentials).get()
    output.eval_js("""new Promise(resolve => {
      const panel = document.createElement('div');
      const text = document.createElement('p');
      text.textContent = 'Copy credentials. In Colab Secrets (key icon), create H3_DRIVE_AUTH, paste the JSON, and enable notebook access. Then click Saved — continue below.';
      const input = document.createElement('input'); input.type = 'password'; input.autocomplete = 'off';
      input.value = AUTH_JSON; input.style.width = '90%';
      const copy = document.createElement('button'); copy.textContent = 'Copy credentials';
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(input.value); copy.textContent = 'Copied'; }
        catch { input.focus(); input.select(); copy.textContent = 'Press Ctrl+C / Cmd+C'; }
      };
      const saved = document.createElement('button'); saved.textContent = 'Saved — continue';
      saved.onclick = () => { input.value = ''; panel.remove(); resolve(true); };
      panel.append(text, input, copy, saved); document.body.append(panel);
    })""".replace("AUTH_JSON", json.dumps(json.dumps(credentials))), timeout_sec=600)
    try:
        saved = validate_credentials(userdata.get(SECRET_NAME))
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        raise DriveAuthError("Save H3_DRIVE_AUTH and enable notebook access, then run this Drive cell again.") from None
    if saved != credentials:
        raise DriveAuthError("H3_DRIVE_AUTH does not match the new credentials. Replace its value and run this Drive cell again.")
    print("[OK] H3_DRIVE_AUTH saved. Continuing Drive setup.")
    return credentials
