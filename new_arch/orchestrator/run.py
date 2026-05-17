"""orchestrator/run.py — управление запусками с порогами GPU и host RAM.

Запускает задачи из jobs.csv последовательно/параллельно с условиями:
  • GPU utilization < --gpu-util-threshold (%);
  • GPU memory used / total < --gpu-mem-threshold (%).
  • Host RAM used / total < --host-mem-threshold (%).
  • Число одновременно running job'ов < --max-running.
Если хотя бы один порог сработал — новые задачи не стартуют.

Работает только на CUDA-машине (через nvidia-ml-py / NVML).

Запущенные процессы НЕ держатся orchestrator'ом — `start_new_session=True`
переводит каждый child в свою process group. Если orchestrator убит/перезапущен,
дочерние процессы продолжают работать. На рестарте orchestrator подхватывает
состояние из process.csv: для running-записей проверяет PID на жизнеспособность.

state-файл `process.csv` обновляется атомарно (write to tmp + rename) каждые
--poll-interval секунд. Колонки:
  job_id, cmd, status, pid, log_path, start_iso, end_iso, exit_code,
  gpu_util_at_start, gpu_mem_pct_at_start
status ∈ {queued, running, done, failed, died}.

Использование:
  PYTHONPATH=. uv run python orchestrator/run.py \
      --jobs orchestrator/jobs.csv \
      --gpu-util-threshold 85 --gpu-mem-threshold 75 --host-mem-threshold 50 \
      --max-running 5 \
      --poll-interval 10

Прерывание (Ctrl-C): orchestrator выходит, дочерние процессы продолжают.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

try:
    import pynvml
except ImportError as e:
    raise SystemExit(
        "nvidia-ml-py (pynvml) не установлен. "
        "Установить: uv add nvidia-ml-py"
    ) from e

ROOT = Path(__file__).resolve().parents[1]

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_DIED = "died"

CSV_FIELDS = [
    "job_id", "cmd", "status", "pid", "log_path",
    "start_iso", "end_iso", "exit_code",
    "gpu_util_at_start", "gpu_mem_pct_at_start",
]


@dataclass
class Job:
    job_id: str
    cmd: str
    status: str = STATUS_QUEUED
    pid: Optional[int] = None
    log_path: Optional[str] = None
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    exit_code: Optional[int] = None
    gpu_util_at_start: Optional[int] = None
    gpu_mem_pct_at_start: Optional[int] = None

    def as_row(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "cmd": self.cmd,
            "status": self.status,
            "pid": "" if self.pid is None else str(self.pid),
            "log_path": self.log_path or "",
            "start_iso": self.start_iso or "",
            "end_iso": self.end_iso or "",
            "exit_code": "" if self.exit_code is None else str(self.exit_code),
            "gpu_util_at_start": "" if self.gpu_util_at_start is None else str(self.gpu_util_at_start),
            "gpu_mem_pct_at_start": "" if self.gpu_mem_pct_at_start is None else str(self.gpu_mem_pct_at_start),
        }


# ─── GPU мониторинг через NVML ──────────────────────────────────────────────

class GpuMonitor:
    def __init__(self, gpu_index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.name, bytes):
            self.name = self.name.decode("utf-8", errors="replace")

    def util_percent(self) -> int:
        return int(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)

    def mem_used_pct(self) -> int:
        m = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return int(round(100 * m.used / max(m.total, 1)))


def host_mem_used_pct() -> int:
    """Доля занятой host RAM по данным /proc/meminfo.

    Используем MemAvailable, потому что она лучше отражает реальный запас памяти
    под новые процессы, чем просто поле MemFree.
    """
    meminfo: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if not parts:
                continue
            meminfo[key] = int(parts[0])
    total_kib = meminfo.get("MemTotal", 0)
    avail_kib = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
    used_kib = max(total_kib - avail_kib, 0)
    return int(round(100 * used_kib / max(total_kib, 1)))


# ─── State I/O ──────────────────────────────────────────────────────────────

def load_state(state_csv: Path) -> dict[str, Job]:
    """Читает process.csv в dict {job_id: Job}. Если файл отсутствует — {}."""
    if not state_csv.exists():
        return {}
    out: dict[str, Job] = {}
    with state_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["job_id"]] = Job(
                job_id=row["job_id"],
                cmd=row["cmd"],
                status=row.get("status") or STATUS_QUEUED,
                pid=int(row["pid"]) if row.get("pid") else None,
                log_path=row.get("log_path") or None,
                start_iso=row.get("start_iso") or None,
                end_iso=row.get("end_iso") or None,
                exit_code=int(row["exit_code"]) if row.get("exit_code") else None,
                gpu_util_at_start=int(row["gpu_util_at_start"]) if row.get("gpu_util_at_start") else None,
                gpu_mem_pct_at_start=int(row["gpu_mem_pct_at_start"]) if row.get("gpu_mem_pct_at_start") else None,
            )
    return out


def write_state_atomic(state_csv: Path, jobs: list[Job]) -> None:
    """Атомарная запись process.csv (write to .tmp, rename)."""
    tmp = state_csv.with_suffix(".csv.tmp")
    state_csv.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for j in jobs:
            w.writerow(j.as_row())
    os.replace(tmp, state_csv)


def load_jobs_csv(jobs_csv: Path) -> list[Job]:
    """Загружает план задач из jobs.csv (только job_id + cmd важны для нас)."""
    with jobs_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        return [Job(job_id=row["job_id"], cmd=row["cmd"]) for row in r]


def merge_with_state(plan: list[Job], state: dict[str, Job]) -> list[Job]:
    """Берёт jobs из plan, если есть в state — поднимает статус из state."""
    out: list[Job] = []
    for j in plan:
        if j.job_id in state:
            s = state[j.job_id]
            # cmd берём из state если совпадает; если нет — план перезаписывает.
            if s.cmd == j.cmd:
                out.append(s)
            else:
                out.append(j)
        else:
            out.append(j)
    # Добавляем jobs из state, которых нет в plan (на всякий).
    plan_ids = {j.job_id for j in plan}
    for jid, s in state.items():
        if jid not in plan_ids:
            out.append(s)
    return out


# ─── PID и launch ────────────────────────────────────────────────────────────

def _proc_state(pid: int) -> Optional[str]:
    """Возвращает однобуквенный State из /proc/PID/status (R/S/D/Z/T/...) или None.

    Нужно, чтобы отличать живой процесс от zombie (`Z`): для zombie
    `os.kill(pid, 0)` всё ещё успешен (запись в process table есть),
    но это уже мёртвый процесс, ожидающий wait() от родителя.
    """
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("State:"):
                    # Формат: 'State:\tZ (zombie)'
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
                    return None
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # процесс существует, просто наш UID не имеет прав
    # Доп. проверка на zombie через /proc (Linux).
    state = _proc_state(pid)
    if state == "Z":
        return False
    return True


def _reap_zombie(pid: int) -> None:
    """Best-effort wait() для зачистки zombie-дочернего процесса.

    Работает только если pid — наш прямой child (запущенный в этой
    сессии оркестратора). После рестарта оркестратора это no-op,
    но zombie всё равно отсеется через _proc_state == 'Z'.
    """
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # не наш child или уже reaped
    except OSError:
        pass


def launch_job(job: Job, log_dir: Path, gpu_util: int, gpu_mem: int) -> None:
    """Запускает job через bash -c, в новой process group, с записью exit code в .exit."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.job_id}.log"
    exit_path = log_dir / f"{job.job_id}.exit"
    if exit_path.exists():
        exit_path.unlink()

    wrapped = f'({job.cmd}); echo $? > "{exit_path}"'
    log_file = log_path.open("w")
    p = subprocess.Popen(
        ["bash", "-c", wrapped],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job.pid = p.pid
    job.status = STATUS_RUNNING
    if log_path.is_absolute():
        job.log_path = str(log_path.relative_to(ROOT))
    else:
        job.log_path = str(log_path)
    job.start_iso = dt.datetime.now().isoformat(timespec="seconds")
    job.gpu_util_at_start = gpu_util
    job.gpu_mem_pct_at_start = gpu_mem


def read_exit_code(job: Job, log_dir: Path) -> Optional[int]:
    exit_path = log_dir / f"{job.job_id}.exit"
    if not exit_path.exists():
        return None
    try:
        return int(exit_path.read_text().strip())
    except (ValueError, OSError):
        return None


def update_running_job(job: Job, log_dir: Path) -> bool:
    """Если процесс задачи мёртв — обновляет статус. Возвращает True если изменилось."""
    if job.status != STATUS_RUNNING or job.pid is None:
        return False
    if pid_alive(job.pid):
        return False
    # Процесс мёртв (или zombie) — зачищаем и определяем результат.
    _reap_zombie(job.pid)
    code = read_exit_code(job, log_dir)
    job.end_iso = dt.datetime.now().isoformat(timespec="seconds")
    if code is None:
        job.status = STATUS_DIED
        job.exit_code = None
    else:
        job.exit_code = code
        job.status = STATUS_DONE if code == 0 else STATUS_FAILED
    return True


# ─── Main loop ──────────────────────────────────────────────────────────────

def summarize(jobs: list[Job]) -> dict[str, int]:
    c = {s: 0 for s in (STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED, STATUS_DIED)}
    for j in jobs:
        c[j.status] = c.get(j.status, 0) + 1
    return c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU-aware orchestrator")
    p.add_argument("--jobs", type=Path,
                   default=Path(__file__).resolve().parent / "jobs.csv")
    p.add_argument("--state", type=Path,
                   default=Path(__file__).resolve().parent / "process.csv")
    p.add_argument("--log-dir", type=Path,
                   default=Path(__file__).resolve().parent / "logs")
    p.add_argument("--gpu-util-threshold", type=int, default=85,
                   help="запускать новые задачи только если GPU util < N%%")
    p.add_argument("--gpu-mem-threshold", type=int, default=75,
                   help="запускать новые задачи только если GPU mem used < N%%")
    p.add_argument("--host-mem-threshold", type=int, default=50,
                   help="запускать новые задачи только если host RAM used < N%%")
    p.add_argument("--max-running", type=int, default=5,
                   help="максимум одновременно running job'ов")
    p.add_argument("--poll-interval", type=int, default=10,
                   help="секунды между опросом GPU и записью state")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"orchestrator: jobs={args.jobs}  state={args.state}  log_dir={args.log_dir}")
    print(f"  пороги: GPU util < {args.gpu_util_threshold}%, "
          f"GPU mem < {args.gpu_mem_threshold}%, "
          f"host RAM < {args.host_mem_threshold}%, "
          f"max_running < {args.max_running}, poll={args.poll_interval}s")

    if not args.jobs.exists():
        raise SystemExit(f"jobs.csv не найден: {args.jobs}")

    monitor = GpuMonitor(args.gpu_index)
    print(f"  GPU[{args.gpu_index}]: {monitor.name}")

    plan = load_jobs_csv(args.jobs)
    state = load_state(args.state)
    jobs = merge_with_state(plan, state)

    # На restart: revive running записи, помечаем died если PID мёртв.
    for j in jobs:
        if j.status == STATUS_RUNNING and j.pid is not None:
            if not pid_alive(j.pid):
                _reap_zombie(j.pid)
                code = read_exit_code(j, args.log_dir)
                j.end_iso = dt.datetime.now().isoformat(timespec="seconds")
                if code is None:
                    j.status = STATUS_DIED
                else:
                    j.exit_code = code
                    j.status = STATUS_DONE if code == 0 else STATUS_FAILED

    write_state_atomic(args.state, jobs)
    summary = summarize(jobs)
    print(f"  старт: queued={summary[STATUS_QUEUED]}  "
          f"running={summary[STATUS_RUNNING]}  done={summary[STATUS_DONE]}  "
          f"failed={summary[STATUS_FAILED]}  died={summary[STATUS_DIED]}")

    # ── Главный цикл ────────────────────────────────────────────────────────
    try:
        while True:
            # 1. Обновляем running записи (живой/мёртвый PID).
            for j in jobs:
                update_running_job(j, args.log_dir)

            # 2. Опрос GPU.
            gpu_util = monitor.util_percent()
            gpu_mem = monitor.mem_used_pct()
            host_mem = host_mem_used_pct()

            # 3. Сколько задач можно запустить ещё?
            queued = [j for j in jobs if j.status == STATUS_QUEUED]
            running = [j for j in jobs if j.status == STATUS_RUNNING]

            # Старт новых: только если оба порога ОК.
            launched_this_tick = 0
            if (len(running) < args.max_running
                and gpu_util < args.gpu_util_threshold
                and gpu_mem < args.gpu_mem_threshold
                and host_mem < args.host_mem_threshold):
                # Запускаем по одной за тик — даём метрикам обновиться.
                if queued:
                    j = queued[0]
                    launch_job(j, args.log_dir, gpu_util, gpu_mem)
                    launched_this_tick = 1
                    print(f"  [{j.job_id}] start  pid={j.pid}  "
                          f"util={gpu_util}%  gpu_mem={gpu_mem}%  "
                          f"ram={host_mem}%  cmd={j.cmd[:60]}...")

            # 4. Save state.
            write_state_atomic(args.state, jobs)

            # 5. Лог сводки.
            s = summarize(jobs)
            print(f"  tick: util={gpu_util:>3d}%  gpu_mem={gpu_mem:>3d}%  "
                  f"ram={host_mem:>3d}%  "
                  f"queued={s[STATUS_QUEUED]}  running={s[STATUS_RUNNING]}  "
                  f"done={s[STATUS_DONE]}  failed={s[STATUS_FAILED]}  "
                  f"died={s[STATUS_DIED]}  +{launched_this_tick}")

            # 6. Условие выхода.
            if s[STATUS_QUEUED] == 0 and s[STATUS_RUNNING] == 0:
                print("orchestrator: очередь пуста, все процессы завершены.")
                break

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\norchestrator: прерван (Ctrl-C). Запущенные процессы продолжают работать.")
        write_state_atomic(args.state, jobs)


if __name__ == "__main__":
    main()
