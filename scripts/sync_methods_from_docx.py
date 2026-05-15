"""Синхронизация disser-ml-text/text_01–06.md с актуальным текстом методов
из Тамразов.docx (разделы 2.7–2.11).

Маппинг (по скаффолдам disser-ml-text/00_obzor_struktury.md):
  text_01_postanovka_zadachi      ← 2.8.1 + 2.8.2
  text_02_priznaki_i_modalnosti   ← 2.7
  text_03_semejstva_modelej       ← 2.8.6 + 2.8.7 + 2.8.8 + 2.8.9
  text_04_validaciya              ← 2.8.3 + 2.8.4 + 2.8.5
  text_05_metriki_i_postobrabotka ← 2.9 + 2.10
  text_06_statobrabotka           ← 2.11

Границы разделов определяются по стилям заголовков (21 = 2.x, 31 = 2.x.y,
41 = 2.x.y.z) и по тексту первой строки. Стили markdown:
  21 → ##   31 → ###   41 → ####   a5 → bullet   обычный → абзац

Запуск:
    uv run python scripts/sync_methods_from_docx.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Тамразов.docx"
OUT = ROOT / "disser-ml-text"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def get_style(p) -> str:
    pPr = p.find(W + "pPr")
    if pPr is None:
        return ""
    ps = pPr.find(W + "pStyle")
    return ps.get(W + "val") if ps is not None else ""


def text_of(p) -> str:
    return "".join(t.text for t in p.iter(W + "t") if t.text)


def is_table_row(p) -> bool:
    """Грубо: стиль начинается с 'a' (aa, ad — табличные)."""
    s = get_style(p)
    return s.startswith("a") and s != "a5"


def para_to_md(p) -> str:
    txt = text_of(p)
    st = get_style(p)
    if not txt.strip():
        return ""
    if st == "21":
        return f"## {txt.strip()}"
    if st == "31":
        return f"### {txt.strip()}"
    if st == "41":
        return f"#### {txt.strip()}"
    if st == "a5":
        return f"- {txt.strip()}"
    return txt.strip()


def is_table_para(p) -> bool:
    s = get_style(p)
    # реальные ячейки таблицы помечаются стилями типа 'ad', 'aa' (бордюр/заголовок)
    return s in ("aa", "ad", "ae", "af") or (s.startswith("a") and s != "a5" and s != "")


def render_table_block(paras: list, start: int) -> tuple[str, int]:
    """Собирает подряд идущие табличные ячейки в markdown-таблицу.
    Возвращает (markdown, индекс следующего НЕ-таблично абзаца).
    Эвристика: ячейки идут подряд; первая строка таблицы — заголовок.
    Группировка по строкам — приближённая: одна строка = один проход стиля 'aa'/'ad'
    одинаковой полосой; для простоты считаем все ячейки до прерывания обычным
    стилем как «плоский» список и склеиваем по N колонок (определяется по первой
    «строке заголовков»).
    """
    cells = []
    i = start
    while i < len(paras) and is_table_para(paras[i]):
        cells.append(text_of(paras[i]).strip())
        i += 1
    # Простейшее решение: вывести как маркированный список ячеек —
    # пользователь руками отформатирует под markdown-table при необходимости.
    md = "\n".join(f"- {c}" for c in cells if c)
    return md, i


def extract_range(paras, start: int, end: int) -> str:
    """Конвертирует диапазон абзацев в markdown-блок."""
    out_lines = []
    i = start
    while i < end and i < len(paras):
        p = paras[i]
        if is_table_para(p):
            md, i = render_table_block(paras, i)
            if md:
                out_lines.append(md)
            continue
        md = para_to_md(p)
        if md:
            out_lines.append(md)
        i += 1
    return "\n\n".join(out_lines)


def find_heading_idx(paras, target_text: str, expected_style: str | None = None) -> int | None:
    """Ищет абзац с указанным началом текста (и опционально — стилем)."""
    for i, p in enumerate(paras):
        txt = text_of(p).strip()
        if txt.startswith(target_text):
            if expected_style is None or get_style(p) == expected_style:
                return i
    return None


def main() -> None:
    z = zipfile.ZipFile(DOCX)
    root = ET.fromstring(z.read("word/document.xml"))
    paras = list(root.iter(W + "p"))

    # Маркеры разделов: фильтр по стилю заголовка (TOC использует 23/33/43,
    # реальные заголовки — 21/31/41; заголовок 1-го уровня = 1 или 2).
    SECTION_STYLES = {
        # 2.X — стиль 21
        **{lab: "21" for lab in ["2.7 ", "2.8 ", "2.9 ", "2.10 ", "2.11 "]},
        # 2.X.Y — стиль 31, но 2.7.7 и 2.8.9 могут быть размечены как 21 (свежие правки)
        **{lab: "31" for lab in [
            "2.7.1", "2.7.2", "2.7.3", "2.7.4", "2.7.5", "2.7.6",
            "2.8.1", "2.8.2", "2.8.3", "2.8.4", "2.8.5", "2.8.6",
            "2.8.7", "2.8.8",
            "2.9.1", "2.9.2", "2.9.3", "2.9.4",
            "2.11.1", "2.11.2", "2.11.3", "2.11.4", "2.11.5",
        ]},
        # 2.7.7 и 2.8.9 имеют стиль 21 (по факту вывода маркеров)
        "2.7.7": "21",
        "2.8.9": "31",
        # Глава — стиль 1 или 2
    }
    marks = {}
    for label, st in SECTION_STYLES.items():
        idx = find_heading_idx(paras, label, expected_style=st)
        if idx is not None:
            marks[label.strip()] = idx
    # Глава 3 — пробуем стили 1 и 2
    for st in ("1", "2"):
        idx = find_heading_idx(paras, "3 ", expected_style=st)
        if idx is not None:
            marks["3"] = idx
            break
    print("Найдены маркеры:")
    for k in sorted(marks):
        print(f"  {k}: абз. {marks[k]}")

    # Диапазоны для каждого text-файла
    mapping = {
        "text_01_postanovka_zadachi.md": ("2.8.1", "2.8.3"),
        "text_02_priznaki_i_modalnosti.md": ("2.7", "2.8"),
        "text_03_semejstva_modelej.md": ("2.8.6", "2.9"),
        "text_04_validaciya.md": ("2.8.3", "2.8.6"),
        "text_05_metriki_i_postobrabotka.md": ("2.9", "2.11"),
        "text_06_statobrabotka.md": ("2.11", "3"),
    }

    header_tpl = "<!-- АВТОСИНХРОНИЗИРОВАНО из Тамразов.docx (разделы {a}–{b}) скриптом scripts/sync_methods_from_docx.py. Ручные правки будут перезаписаны при следующей синхронизации. -->\n\n"

    for fname, (start_label, end_label) in mapping.items():
        si = marks.get(start_label)
        ei = marks.get(end_label)
        if si is None or ei is None:
            print(f"  [skip] {fname}: маркер не найден ({start_label} → {end_label})")
            continue
        body = extract_range(paras, si, ei)
        full = header_tpl.format(a=start_label, b=end_label) + body + "\n"
        out_path = OUT / fname
        out_path.write_text(full, encoding="utf-8")
        print(f"  ✓ {fname}  ({ei - si} абзацев, {len(body)} символов)")


if __name__ == "__main__":
    main()
