#!/usr/bin/env python3
"""Log what startup is waiting on: network, disk writes, dirty pages, CPU and stall pressure.

run_h3.sh starts it in the background (stdlib only, Linux /proc). Every interval it prints
one ``[h3-stats]`` line; it stops once the watched processes have exited and ComfyUI
answers, or after ``--max-seconds``. Turn it off with H3_STARTUP_STATS=0.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import urllib.request


def _net_rx() -> int:
    total = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, data = line.split(":", 1)
        if name.strip() != "lo":
            total += int(data.split()[0])
    return total


def _disk_written() -> dict[str, int]:
    """Bytes written per physical disk: /sys/block lists no partitions; loop, RAM and
    device-mapper devices are left out so nothing counts twice."""
    disks = {name for name in os.listdir("/sys/block") if not name.startswith(("loop", "ram", "zram", "dm-"))}
    found = {}
    for line in Path("/proc/diskstats").read_text().splitlines():
        parts = line.split()
        if parts[2] in disks:
            found[parts[2]] = int(parts[9]) * 512
    return found


def _meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) * 1024
    return values


def _cpu() -> tuple[int, int, int]:
    """(busy, iowait, total) jiffies."""
    fields = [int(value) for value in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    user, nice, system, idle, iowait, irq, softirq, steal = (fields + [0] * 8)[:8]
    busy = user + nice + system + irq + softirq + steal
    return busy, iowait, busy + idle + iowait


def _pressure(kind: str) -> str:
    """``some/full`` avg10 stall percentages from PSI, or "n/a" without it."""
    try:
        lines = Path(f"/proc/pressure/{kind}").read_text().splitlines()
    except OSError:
        return "n/a"
    shares = {}
    for line in lines:
        name, *pairs = line.split()
        shares[name] = dict(pair.split("=") for pair in pairs)["avg10"]
    return "/".join(f"{shares[name]}%" for name in ("some", "full") if name in shares)


def _alive(pids: list[int], commands: list[str] = ()) -> bool:
    """Whether any of ``pids``, or any process whose command line contains one of ``commands``, runs."""
    for pid in pids:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            continue
    if not commands:
        return False
    own = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        try:
            line = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(command in line for command in commands):
            return True
    return False


def _answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status < 500
    except Exception:
        return False


def _setting(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--watch-pid", type=int, action="append", default=[])
    # Processes started after this one (run_h3.sh's later h3_prepare.py) are found by command line.
    parser.add_argument("--watch-command", action="append", default=[])
    parser.add_argument("--until-url", default="")
    args = parser.parse_args(argv)
    if not Path("/proc/stat").is_file():
        return 0
    memory = _meminfo()
    print(f"[h3-stats] RAM {memory.get('MemTotal', 0) / 2**30:.0f} GiB; dirty_ratio "
          f"{_setting('/proc/sys/vm/dirty_ratio')}% background {_setting('/proc/sys/vm/dirty_background_ratio')}%; "
          f"CPUs {os.cpu_count()}; disks {', '.join(sorted(_disk_written())) or 'none'}", flush=True)
    started = time.monotonic()
    net, disk, cpu = _net_rx(), _disk_written(), _cpu()
    while time.monotonic() - started < args.max_seconds:
        time.sleep(args.interval)
        now_net, now_disk, now_cpu, memory = _net_rx(), _disk_written(), _cpu(), _meminfo()
        written = sum(now_disk.get(name, 0) - disk.get(name, 0) for name in now_disk)
        busy, iowait, total = (now - before for now, before in zip(now_cpu, cpu))
        total = max(1, total)
        print(f"[h3-stats] +{time.monotonic() - started:.0f}s net {(now_net - net) * 8 / args.interval / 1e9:.1f} Gb/s"
              f" | disk write {written / args.interval / 1e6:.0f} MB/s"
              f" | dirty {memory.get('Dirty', 0) / 1e9:.1f} GB writeback {memory.get('Writeback', 0) / 1e9:.1f} GB"
              f" | cpu {100 * busy / total:.0f}% iowait {100 * iowait / total:.0f}%"
              f" | stalled io {_pressure('io')} cpu {_pressure('cpu')} mem {_pressure('memory')}", flush=True)
        net, disk, cpu = now_net, now_disk, now_cpu
        if not _alive(args.watch_pid, args.watch_command) and (not args.until_url or _answers(args.until_url)):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
