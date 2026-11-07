# Disposable engineering diagnostic. A positive probe is evidence of an
# exposed channel, but negative probes do not prove a hermetic build boundary.
{ hostProbe, workspaceProbe, loopbackPort }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  pkgs = import (builtins.fetchTree lock.nodes.nixpkgs.locked) { system = "aarch64-darwin"; };
in
pkgs.runCommand "antecedent-sandbox-probe" {
  nativeBuildInputs = [ pkgs.python3 ];
  TEST_HOST_FILE = hostProbe;
  TEST_WORKSPACE_FILE = workspaceProbe;
  TEST_LOOPBACK_PORT = loopbackPort;
} ''
  python3 - <<'PY'
  import json, os, socket, subprocess, sys, time
  from pathlib import Path

  def probe_file(name):
      try:
          with open(os.environ[name], 'rb') as probe:
              probe.read(1)
          return 'accessible'
      except PermissionError:
          return 'denied'
      except FileNotFoundError:
          return 'hidden'
      except OSError as error:
          return 'error:' + str(error.errno)

  sock = socket.socket()
  sock.settimeout(2)
  try:
      sock.connect(('127.0.0.1', int(os.environ['TEST_LOOPBACK_PORT'])))
      network = 'accessible'
  except PermissionError:
      network = 'denied'
  except OSError as error:
      network = 'error:' + str(error.errno)
  finally:
      sock.close()

  # Probe whether a nested sandbox can additionally deny the accessible
  # sentinel. Success for this one path is not proof for other host inputs.
  profile = '(version 1) (allow default) (deny file-read* (literal ' + json.dumps(os.environ['TEST_HOST_FILE']) + '))'
  child = 'import os; open(os.environ["TEST_HOST_FILE"], "rb").read(1)'
  try:
      nested = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, sys.executable, '-c', child],
                              capture_output=True, text=True, timeout=10)
      if nested.returncode == 0:
          nested_result = 'file_accessible'
      elif 'Operation not permitted' in nested.stderr and 'sandbox_apply' in nested.stderr:
          nested_result = 'sandbox_apply_denied'
      elif 'PermissionError' in nested.stderr:
          nested_result = 'file_denied'
      else:
          nested_result = 'error_exit:' + str(nested.returncode)
  except OSError as error:
      nested_result = 'error:' + str(error.errno)
  except subprocess.TimeoutExpired:
      nested_result = 'timeout'

  # Clock and entropy are channels even if files and network are isolated.
  result = {
      'private_tmp_file': probe_file('TEST_HOST_FILE'),
      'workspace_file': probe_file('TEST_WORKSPACE_FILE'),
      'loopback': network,
      'nested_sandbox_file': nested_result,
      'clock_available': time.time_ns() > 0,
      'os_entropy_available': len(os.urandom(8)) == 8,
  }
  Path(os.environ['out']).write_text(json.dumps(result, sort_keys=True))
  PY
''
