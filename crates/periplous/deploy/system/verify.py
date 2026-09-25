#!/usr/bin/env python3
"""Runs as the dedicated prod identity under the exact production restrictions."""
import errno
import json
import os
from pathlib import Path
import socket
import subprocess

root = Path('/var/lib/periplous')
operator = json.loads(Path('/etc/periplous/operator.json').read_text())
assert os.getuid() not in (0, operator['uid']), 'prod must have its own identity'
assert os.getgroups() == [os.getgid()], 'unexpected supplementary groups'
status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert status['NoNewPrivs'].strip() == '1'
assert int(status['CapEff'].strip(), 16) == 0
assert status['Seccomp'].strip() == '2', 'syscall filter must actually be enforced'
root_mount = next(line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4] == '/')
assert 'ro' in root_mount[5].split(','), 'root filesystem is not read-only'
assert os.readlink('/proc/self/ns/mnt') != operator['host_mount_namespace'], 'mount isolation is ineffective'
try:
    os.listdir(operator['home'])
except PermissionError:
    pass
else:
    raise AssertionError('operator home is accessible')
try:
    descriptor = os.open(root / 'protection-canary', os.O_WRONLY)
except OSError as error:
    assert error.errno in (errno.EACCES, errno.EROFS)
else:
    os.close(descriptor)
    raise AssertionError('prod can write deployment-owned files')
try:
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(1)
        connection.connect(f"/run/user/{operator['uid']}/systemd/private")
except OSError:
    pass
else:
    raise AssertionError('operator user-manager socket is accessible')
state = json.loads((root / 'environments/prod/state.json').read_text())
binary = root / 'releases' / state['current'] / 'periplous'
snapshot = json.loads(subprocess.check_output([binary, 'snapshot'], timeout=15))
assert snapshot['host']['cpu'] is not None and snapshot['host']['memory'] is not None
assert len(snapshot['gpus'] or []) == 2, 'both Skylake GPUs must remain observable'
for gpu in snapshot['gpus']:
    assert gpu['memory'] is not None and gpu['utilization_percent'] is not None
assert not snapshot['issues'], snapshot['issues']
print(json.dumps({'uid': os.getuid(), 'gid': os.getgid(), 'groups': os.getgroups(),
                  'mount_namespace': os.readlink('/proc/self/ns/mnt'), 'root_mount_readonly': True,
                  'home_blocked': True, 'deployment_writes_blocked': True,
                  'user_manager_blocked': True, 'no_new_privileges': True,
                  'seccomp_enforced': True, 'capabilities_empty': True,
                  'gpus': len(snapshot['gpus']), 'telemetry_issues': snapshot['issues']}, indent=2))
