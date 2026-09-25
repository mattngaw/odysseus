#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help && $# == 1 ]]; then
    echo 'Usage: bash crates/periplous/deploy/install.sh'
    echo 'Install controller and units only. Does not deploy, restart, enable, or publish.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Installer takes no arguments; use periplousctl stage/deploy.' >&2; exit 2; }
[[ $(uname -s) == Linux && $(id -u) != 0 ]] || { echo 'Run as the non-root Linux service user.' >&2; exit 1; }
for tool in systemctl python3 install mktemp; do command -v "$tool" >/dev/null; done
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_dir="$HOME/.local/share/periplous"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
for unit in periplous@.service periplous-tunnel.service; do
    if [[ -e $unit_dir/$unit ]] && ! grep -q '^# Managed by crates/periplous/deploy/install.sh$' "$unit_dir/$unit"; then
        echo "Refusing to replace unmanaged unit: $unit_dir/$unit" >&2
        exit 1
    fi
done
install -d -m 700 "$data_dir"
install -d "$data_dir/bin" "$unit_dir" "$HOME/.local/bin"
staged=''
trap 'if [[ -n $staged ]]; then rm -f -- "$staged"; fi' EXIT
atomic_install() {
    local mode=$1 source=$2 destination=$3
    staged=$(mktemp "$(dirname -- "$destination")/.periplous-install.XXXXXX")
    install -m "$mode" -- "$source" "$staged"
    mv -f -- "$staged" "$destination"
    staged=''
}
atomic_install 755 "$source_dir/periplousctl" "$HOME/.local/bin/periplousctl"
atomic_install 644 "$source_dir/quick-tunnel.yml" "$data_dir/quick-tunnel.yml"
for unit in periplous@.service periplous-tunnel.service; do
    atomic_install 644 "$source_dir/$unit" "$unit_dir/$unit"
done
# Generic defaults are loopback. Operators configure interfaces explicitly.
for env in dev prod; do
    install -d -m 700 "$data_dir/environments/$env"
    if [[ ! -e $data_dir/environments/$env/config.json ]]; then
        port=8765
        [[ $env == prod ]] || port=8766
        printf '{"bind":"127.0.0.1","port":%s}\n' "$port" > "$data_dir/environments/$env/config.json"
    fi
done
if cloudflared=$(command -v cloudflared); then
    ln -sfn -- "$(readlink -f -- "$cloudflared")" "$data_dir/bin/cloudflared"
fi
systemctl --user daemon-reload
echo 'Installed. Stage a release, then deploy dev and promote after review.'
echo 'Legacy periplous-dashboard.service is not stopped automatically; migrate it explicitly.'
