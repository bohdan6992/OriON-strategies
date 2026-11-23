"""Робить запис статусу раннера (latest.json / latest.md) безпечним.

Навіщо: раніше помилка запису latest.md (OSError [Errno 22]) валила весь раннер посеред прогону:
CRACEN лишався сиротою, а стратегії не запускались. Після патчу запис статусу атомарний
(tmp + os.replace, 5 спроб) і ніколи не кидає виняток.

Запуск (безпечно повторювати; перед змінами робить run_orion_daily.py.bak):
    python patch_runner.py [шлях до run_orion_daily.py]
"""
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\datum-api-examples-main\OriON\run_orion_daily.py")
src = target.read_text(encoding="utf-8")

if "_safe_write_text" in src:
    print("вже запатчено:", target)
    sys.exit(0)

HELPER = '''def _safe_write_text(path, text):
    """Атомарний запис (tmp + os.replace) із повторами; помилка запису статусу НІКОЛИ не валить раннер."""
    import os, time as _t
    last = None
    for attempt in range(5):
        try:
            tmp = path.with_name(path.name + ".tmp_write")
            tmp.write_text(text, encoding="utf-8")
            os.replace(str(tmp), str(path))
            return True
        except Exception as ex:
            last = ex
            _t.sleep(0.2 * (attempt + 1))
    try:
        print(f"[status] не вдалось записати {path.name}: {last} (ігнорую, прогін триває)", flush=True)
    except Exception:
        pass
    return False


'''

pairs = [
    ('def write_status(orion_home: Path, status: dict):', HELPER + 'def write_status(orion_home: Path, status: dict):'),
    (r'    (st_dir / "latest.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")',
     r'    _safe_write_text(st_dir / "latest.json", json.dumps(status, indent=2, ensure_ascii=False) + "\n")'),
    (r'    (st_dir / "latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")',
     r'    _safe_write_text(st_dir / "latest.md", "\n".join(md) + "\n")'),
]
for old, new in pairs:
    if src.count(old) != 1:
        sys.exit(f"НЕ ПАТЧУ: очікуваний фрагмент не знайдено рівно 1 раз: {old[:70]!r} (знайдено {src.count(old)})")
    src = src.replace(old, new)

compile(src, str(target), "exec")          # синтаксична перевірка до запису
shutil.copy2(target, str(target) + ".bak")
target.write_text(src, encoding="utf-8")
print("запатчено:", target, "(резервна копія: .bak)")
