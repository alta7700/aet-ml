"""Оркестратор параллельных процессов для управления пулом обучений.

Переиспользуемый модуль для запуска множества независимых задач
параллельно с контролем количества одновременных процессов.
"""

from __future__ import annotations

import subprocess
import json
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, asdict
import time
import sys


@dataclass
class TaskSpec:
    """Спецификация одной задачи для запуска."""
    task_id: str
    script: Path
    args: list[str]
    kwargs: dict[str, Any] | None = None


class ProcessPoolOrchestrator:
    """Оркестратор пула процессов с контролем нагрузки.

    Поддерживает пул из max_workers активных процессов.
    Автоматически запускает новые задачи когда освобождаются слоты.
    """

    def __init__(self, max_workers: int = 4, verbose: bool = True):
        """
        Параметры
        ----------
        max_workers : int
            Максимальное количество одновременных процессов
        verbose : bool
            Выводить информацию о прогрессе
        """
        self.max_workers = max_workers
        self.verbose = verbose
        self.active_processes: dict[str, subprocess.Popen] = {}
        self.completed_tasks: list[str] = []
        self.failed_tasks: list[tuple[str, str]] = []
        self.results: dict[str, Any] = {}

    def submit_tasks(self, tasks: list[TaskSpec]) -> None:
        """Запускает все задачи с контролем нагрузки."""
        task_queue = list(tasks)
        task_idx = 0

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"ОРКЕСТРАТОР: {len(tasks)} задач, max_workers={self.max_workers}")
            print(f"{'='*70}\n")

        while task_idx < len(task_queue) or self.active_processes:
            # Запускаем новые задачи если есть свободные слоты
            while task_idx < len(task_queue) and \
                  len(self.active_processes) < self.max_workers:
                task = task_queue[task_idx]
                self._start_task(task)
                task_idx += 1

            # Проверяем завершенные процессы
            if self.active_processes:
                self._check_processes()
                time.sleep(1)

    def _start_task(self, task: TaskSpec) -> None:
        """Запускает один процесс для задачи."""
        cmd = [
            sys.executable,
            str(task.script),
            *task.args,
        ]

        if task.kwargs:
            for k, v in task.kwargs.items():
                cmd.extend([f"--{k}", str(v)])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.active_processes[task.task_id] = proc

            if self.verbose:
                active_count = len(self.active_processes)
                print(f"  [{active_count}/{self.max_workers}] Запущена: {task.task_id}")

        except Exception as e:
            self.failed_tasks.append((task.task_id, str(e)))
            if self.verbose:
                print(f"  ❌ Ошибка запуска {task.task_id}: {e}")

    def _check_processes(self) -> None:
        """Проверяет статус активных процессов."""
        finished = []

        for task_id, proc in list(self.active_processes.items()):
            poll_result = proc.poll()

            if poll_result is not None:  # Процесс завершился
                finished.append(task_id)

                if poll_result == 0:
                    self.completed_tasks.append(task_id)
                    if self.verbose:
                        print(f"  ✅ {task_id}")
                else:
                    stdout, _ = proc.communicate()
                    error_msg = stdout[-500:] if stdout else "unknown error"
                    self.failed_tasks.append((task_id, error_msg))
                    if self.verbose:
                        print(f"  ⚠️  {task_id} (код {poll_result})")

                del self.active_processes[task_id]

    def wait_all(self) -> dict[str, Any]:
        """Ждёт завершения всех процессов, возвращает результаты."""
        for proc in self.active_processes.values():
            try:
                proc.wait()
            except:
                pass

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"РЕЗУЛЬТАТЫ: {len(self.completed_tasks)} успешно, "
                  f"{len(self.failed_tasks)} ошибок")
            print(f"{'='*70}\n")

            if self.failed_tasks:
                print("Ошибки:")
                for task_id, error in self.failed_tasks:
                    print(f"  ❌ {task_id}")

        return {
            "completed": self.completed_tasks,
            "failed": self.failed_tasks,
            "results": self.results,
        }


def run_task_in_subprocess(task_id: str,
                          task_fn: Callable,
                          *args,
                          **kwargs) -> None:
    """Вспомогательная функция для запуска task_fn внутри subprocess.

    Используется когда нужно запустить Python функцию в отдельном процессе.
    """
    try:
        result = task_fn(task_id, *args, **kwargs)
        # Результат выводим в JSON для парсинга
        print(json.dumps({"status": "ok", "task_id": task_id, "result": result}))
    except Exception as e:
        print(json.dumps({"status": "error", "task_id": task_id, "error": str(e)}))
        sys.exit(1)
