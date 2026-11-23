"""Запускає стратегії раннера ПАРАЛЕЛЬНО (за замовчуванням 4 одночасно) замість по черзі.

Навіщо: усі ~20 стратегій по черзі читають final.parquet (~1 млрд рядків) Python-циклом, один процес = одне
ядро. Прогін триває ~12+ год. Стратегії незалежні (кожна пише лише у свій signals\\<код> і свої stage-каталоги),
тому пул із N процесів дає майже лінійне прискорення (обмежене найдовшою - ArbitRage).

Що робить патч (run_orion_daily.py, виконується лише нова копія при НАСТУПНОМУ запуску):
  * замінює послідовний цикл стратегій на ThreadPoolExecutor(ORION_PARALLEL, за замовч. 4);
  * найбільші ноутбуки стартують першими (найдовші не лишаються "хвостом");
  * запис статусу під замком; у latest.json/md current_notebook = перелік активних через " + ";
  * збій однієї стратегії, як і раніше, не зупиняє інші.
Змінити паралельність: змінна середовища ORION_PARALLEL (1 = як раніше, по черзі).

Запуск (безпечно повторювати; перед змінами робить run_orion_daily.py.bak_parallel):
    python patch_runner_parallel.py [шлях до run_orion_daily.py]
"""
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\datum-api-examples-main\OriON\run_orion_daily.py")
src = target.read_text(encoding="utf-8")

if "ORION_PARALLEL" in src:
    print("вже запатчено:", target)
    sys.exit(0)

START = '        for nb in sorted(notebooks_dir.glob("*.ipynb")):\n            if nb.name == CRACEN_NOTEBOOK:\n                continue\n'
END = '''            update_run("completed" if code == 0 else "failed", name, elapsed_sec=round(info["duration_sec"], 1), last_output=info.get("error", status.get("run", {}).get("last_output", "")))
'''
i = src.find(START)
j = src.find(END, i)
if src.count(START) != 1 or j < 0:
    sys.exit("НЕ ПАТЧУ: блок циклу стратегій не знайдено в очікуваному вигляді (файл відрізняється від очікуваного)")
j += len(END)

NEW = '''        # --- паралельний запуск стратегій (patch_runner_parallel.py) ---
        from concurrent.futures import ThreadPoolExecutor
        _nbs = [nb for nb in notebooks_dir.glob("*.ipynb") if nb.name != CRACEN_NOTEBOOK]
        _nbs.sort(key=lambda p: p.stat().st_size, reverse=True)       # найбільші (найдовші) - першими
        try:
            _workers = max(1, int(os.environ.get("ORION_PARALLEL", "4")))
        except ValueError:
            _workers = 4
        _lock = threading.Lock()
        _active = {}
        print(f"[strategies] {len(_nbs)} ноутбуків, паралельно: {_workers}", flush=True)

        def _run_one(nb):
            name = nb.stem
            t_s = time.time()
            out_nb = str(orion_home / STATUS_DIRNAME / f"last_{name}_out.ipynb")
            with _lock:
                _active[name] = 0.0
                update_run("running", " + ".join(_active), elapsed_sec=0.0, out_notebook=out_nb, last_output="")

            def _progress(snap, notebook_name=name):
                with _lock:
                    _active[notebook_name] = snap.get("elapsed_sec") or 0.0
                    update_run(
                        snap.get("phase") or "running",
                        " + ".join(_active),
                        elapsed_sec=max(_active.values()),
                        started_at_utc=snap.get("started_at_utc"),
                        out_notebook=snap.get("out_notebook"),
                        out_notebook_size=snap.get("out_notebook_size"),
                        out_notebook_mtime_utc=snap.get("out_notebook_mtime_utc"),
                        last_output=f"[{notebook_name}] " + (snap.get("last_output", "") or ""),
                    )

            try:
                code, so, se = run_notebook(nb, cwd=orion_home, env=env, on_progress=_progress)
            except Exception as ex:
                code, so, se = 1, "", f"runner exception: {ex}"
            info = {"ok": (code == 0), "duration_sec": time.time() - t_s}
            if code != 0:
                info["error"] = (se.strip() or so.strip() or f"exit {code}")[:2000]
            with _lock:
                status["strategies"][name] = info
                _active.pop(name, None)
                update_run("running" if _active else ("completed" if code == 0 else "failed"),
                           " + ".join(_active) if _active else name,
                           elapsed_sec=round(info["duration_sec"], 1),
                           last_output=info.get("error", f"[{name}] done"))

        with ThreadPoolExecutor(max_workers=_workers) as _pool:
            list(_pool.map(_run_one, _nbs))
'''
src = src[:i] + NEW + src[j:]

head = src.split("def ")[0]
if "import os" not in head or "import threading" not in head:
    sys.exit("НЕ ПАТЧУ: у run_orion_daily.py немає import os/threading нагорі")
compile(src, str(target), "exec")          # синтаксична перевірка до запису
shutil.copy2(target, str(target) + ".bak_parallel")
target.write_text(src, encoding="utf-8")
print("запатчено:", target, "(резервна копія: .bak_parallel)")
