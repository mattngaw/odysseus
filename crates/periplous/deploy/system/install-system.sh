#!/usr/bin/env bash
set -euo pipefail
usage() {
    echo 'Usage: sudo bash install-system.sh OPERATOR RELEASE_DIRECTORY'
    echo 'Install and verify isolated prod, then migrate port 8765. Leaves the public tunnel stopped.'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# == 2 && $(id -u) == 0 && $(uname -s) == Linux ]] || { usage >&2; exit 2; }
operator=$1
package=$(realpath -- "$2")
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
operator_uid=$(id -u "$operator")
[[ $operator_uid != 0 ]] || { echo 'Operator must not be root.' >&2; exit 1; }
operator_home=$(getent passwd "$operator" | cut -d: -f6)
[[ -d $operator_home && -d $package ]] || exit 1
[[ ! -e /etc/periplous/system-prod ]] || { echo 'Already migrated. Use periplousctl promote for later releases.' >&2; exit 1; }
for path in /var/lib/periplous /etc/periplous /usr/local/libexec/periplous; do
    [[ ! -L $path ]] || { echo "Refusing symlink: $path" >&2; exit 1; }
    if [[ -e $path && $(stat -c %u "$path") != 0 ]]; then
        echo "Refusing non-root-owned directory: $path" >&2; exit 1
    fi
done
for unit in periplous-prod.service periplous-prod.socket periplous-tunnel.service; do
    if [[ -e /etc/systemd/system/$unit ]] && ! grep -q '^# Managed by crates/periplous/deploy/system/install-system.sh$' "/etc/systemd/system/$unit"; then
        echo "Refusing unmanaged unit: $unit" >&2; exit 1
    fi
done
if getent passwd periplous >/dev/null; then
    [[ $(id -u periplous) -lt 1000 && $(getent passwd periplous | cut -d: -f7) == /usr/sbin/nologin ]] || { echo 'Existing periplous identity is not a dedicated system account.' >&2; exit 1; }
else
    useradd --system --user-group --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin periplous
fi
[[ $(id -G periplous) == "$(id -g periplous)" ]] || { echo 'Remove unexpected supplementary groups from periplous first.' >&2; exit 1; }
install -d -m 755 /usr/local/libexec/periplous /var/lib/periplous /var/lib/periplous/releases /var/lib/periplous/environments /var/lib/periplous/environments/prod /etc/periplous
install -m 755 "$source_dir/../periplousctl" /usr/local/libexec/periplous/periplousctl
install -m 755 "$source_dir/prodctl" /usr/local/libexec/periplous/prodctl
install -m 644 "$source_dir/verify.py" /usr/local/libexec/periplous/verify.py
install -m 644 "$source_dir/../quick-tunnel.yml" /etc/periplous/quick-tunnel.yml
for unit in periplous-prod.service periplous-prod.socket periplous-tunnel.service; do
    install -m 644 "$source_dir/$unit" "/etc/systemd/system/$unit"
done
printf 'read-only canary\n' > /var/lib/periplous/protection-canary
chmod 644 /var/lib/periplous/protection-canary
python3 - "$operator" "$operator_uid" "$operator_home" <<'PY'
import json,os,socket,subprocess,sys
from pathlib import Path
operator, uid, home=sys.argv[1:]
Path('/etc/periplous/operator.json').write_text(json.dumps({'user':operator,'uid':int(uid),'home':home,'host_mount_namespace':os.readlink('/proc/self/ns/mnt')})+'\n')
hosts={'localhost','127.0.0.1',socket.gethostname().lower(),'skylake.local'}
for interface in json.loads(subprocess.check_output(['ip','-j','-4','address','show'],text=True)):
    hosts.update(address['local'] for address in interface.get('addr_info',[]) if address['family']=='inet')
Path('/var/lib/periplous/environments/prod/config.json').write_text(json.dumps({'bind':'0.0.0.0','port':8765,'socket_activation':True,'allowed_hosts':sorted(hosts)})+'\n')
PY
chmod 644 /etc/periplous/operator.json /var/lib/periplous/environments/prod/config.json
helper=/usr/local/libexec/periplous/prodctl
if [[ -e /var/lib/periplous/environments/prod/state.json ]]; then
    python3 - "$package" <<'PY'
import json,sys
from pathlib import Path
state=json.loads(Path('/var/lib/periplous/environments/prod/state.json').read_text())
assert state['current']==Path(sys.argv[1]).name and 'pending' not in state, 'partial installation uses a different release'
PY
else
    "$helper" bootstrap "$package"
fi
# Clone the exact service restrictions, changing only lifecycle/command for the
# one-shot negative canary and hardware-collection verification. Prod is untouched.
python3 - <<'PY'
from pathlib import Path
source=Path('/etc/systemd/system/periplous-prod.service').read_text().split('[Service]\n',1)[1]
source=source.replace('Type=exec','Type=oneshot').replace('Restart=on-failure','Restart=no').replace('ExecStart=/usr/local/libexec/periplous/prodctl launch','ExecStart=/usr/bin/python3 /usr/local/libexec/periplous/verify.py')
Path('/run/systemd/system/periplous-prod-verify.service').write_text('[Unit]\nDescription=Verify Periplous prod isolation and telemetry\n\n[Service]\n'+source)
PY
systemctl daemon-reload
systemd-analyze verify /etc/systemd/system/periplous-prod.service /etc/systemd/system/periplous-prod.socket /etc/systemd/system/periplous-tunnel.service
if ! systemctl start periplous-prod-verify.service; then
    journalctl -u periplous-prod-verify.service --lines=80 --no-pager
    echo 'Isolation/telemetry validation failed; existing user prod was not stopped.' >&2
    exit 1
fi
invocation=$(systemctl show periplous-prod-verify.service --property=InvocationID --value)
journalctl "_SYSTEMD_INVOCATION_ID=$invocation" --output=cat --no-pager > /var/lib/periplous/isolation-verification.log
cat /var/lib/periplous/isolation-verification.log
rm /run/systemd/system/periplous-prod-verify.service
systemctl daemon-reload
user_systemctl() {
    runuser -u "$operator" -- env XDG_RUNTIME_DIR="/run/user/$operator_uid" systemctl --user "$@"
}
# Preserve and update the actual operator entry point; the root helper alone
# would leave an older user controller managing the retired prod instance.
user_controller="$operator_home/.local/bin/periplousctl"
user_tunnel="$operator_home/.config/systemd/user/periplous-tunnel.service"
[[ -f $user_controller && -f $user_tunnel ]] || { echo 'Expected the existing user deployment.' >&2; exit 1; }
grep -q '^# Managed by crates/periplous/deploy/install.sh$' "$user_tunnel" || { echo 'Refusing unmanaged user tunnel unit.' >&2; exit 1; }
install -m 600 "$user_controller" /var/lib/periplous/operator-controller-before-system
install -m 600 "$user_tunnel" /var/lib/periplous/operator-tunnel-before-system
# Keep the old user installation available if the actual cutover fails.
restore_user_prod() {
    rm -f /etc/periplous/system-prod
    systemctl stop periplous-tunnel.service periplous-prod.socket periplous-prod.service || true
    systemctl disable periplous-prod.socket || true
    install -o "$operator" -g "$(id -gn "$operator")" -m 755 /var/lib/periplous/operator-controller-before-system "$user_controller"
    install -o "$operator" -g "$(id -gn "$operator")" -m 644 /var/lib/periplous/operator-tunnel-before-system "$user_tunnel"
    user_systemctl daemon-reload
    user_systemctl unmask periplous@prod.service || true
    user_systemctl enable --now periplous@prod.service
    echo 'Cutover failed; restored the previous user prod service. Public tunnel remains stopped.' >&2
}
trap restore_user_prod ERR
user_systemctl stop periplous-tunnel.service periplous@prod.service
install -o "$operator" -g "$(id -gn "$operator")" -m 755 "$source_dir/../periplousctl" "$user_controller"
install -o "$operator" -g "$(id -gn "$operator")" -m 644 "$source_dir/../periplous-tunnel.service" "$user_tunnel"
cmp "$user_controller" /usr/local/libexec/periplous/periplousctl
user_systemctl daemon-reload
systemctl reset-failed periplous-prod.service periplous-prod.socket || true
"$helper" start
# Validate the real socket-activated endpoint before making migration persistent.
python3 - <<'PY'
import json,subprocess,urllib.request
from pathlib import Path
request=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with request.open('http://127.0.0.1:8765/api/deployment',timeout=5) as response: live=json.load(response)
state=json.loads(Path('/var/lib/periplous/environments/prod/state.json').read_text())
assert live=={'environment':'prod','release':state['current']}
pid=subprocess.check_output(['systemctl','show','periplous-prod.service','-p','MainPID','--value'],text=True).strip()
os_user = next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('Uid:'))
assert os_user.split()[1] == subprocess.check_output(['id','-u','periplous'],text=True).strip()
PY
user_systemctl disable periplous@prod.service
user_systemctl mask periplous@prod.service
systemctl enable periplous-prod.socket
printf 'system prod managed by /usr/local/libexec/periplous/prodctl\n' > /etc/periplous/system-prod
chmod 644 /etc/periplous/system-prod
trap - ERR
"$helper" status
! systemctl is-active --quiet periplous-tunnel.service
printf 'Migration complete. Prod is isolated; the public tunnel remains stopped.\n'
