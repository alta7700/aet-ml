# Параметры испытуемых: источник, маппинг, производные индексы

> Файл: `preprocessing/subject_params.md`
> Источник: атрибуты корневой группы `finaltest.h5`
> Выходной файл: `dataset/subjects.parquet`

---

## Источник данных

Все антропометрические и биоимпедансные параметры записываются как атрибуты корневой группы HDF5 при подготовке файла сессии. Биоимпеданс выполнялся перед тестом (биоимпедансный анализатор состава тела).

---

## Маппинг HDF5 → subjects.parquet

| HDF5-атрибут | Колонка в parquet | Тип | Единица | Описание |
|---|---|---|---|---|
| `subject_id` | `subject_id` | str | — | Идентификатор (S001–S014) |
| `subject_name` | `subject_name` | str | — | ФИО |
| `sex` | `sex` | str | — | Пол (`Мужской` / `Женский`) |
| `age` | `age` | int | лет | Возраст |
| `height` | `height` | float | см | Рост |
| `weight` | `weight` | float | кг | Масса тела |
| `body_fat_mass` | `body_fat_mass` | float | кг | Масса жира в теле (биоимпеданс) |
| `skeletal_muscle_mass` | `skeletal_muscle_mass` | float | кг | Масса скелетной мускулатуры |
| `dominant_leg_lean_mass` | `dominant_leg_lean_mass` | float | кг | Тощая масса ведущей ноги |
| `dominant_leg_fat_mass` | `dominant_leg_fat_mass` | float | кг | Жировая масса ведущей ноги |
| `dominant_leg_circumference` | `dominant_leg_circumference` | float | см | Обхват ведущей ноги |
| `phase_angle` | `phase_angle` | float | ° | Фазовый угол биоимпеданса |
| `stop_time_sec` | `stop_time_sec` | float | с | Время остановки теста |

Поля с маркером пропуска (`-`, `—`, `nan`) → `None` в parquet. Для каждого поля из `SUMMARY_FIELDS` автоматически создаётся колонка `{field}_is_missing` (0/1).

---

## Производные индексы (вычисляются в `subjects.py`)

| Колонка | Формула | Физический смысл |
|---|---|---|
| `bmi` | `weight / (height/100)²` | Индекс массы тела |
| `body_fat_pct` | `body_fat_mass / weight × 100` | % жировой массы тела |
| `muscle_to_fat_total` | `skeletal_muscle_mass / body_fat_mass` | Отношение мышц к жиру (всё тело) |
| `leg_fat_pct` | `leg_fat / (leg_lean + leg_fat) × 100` | % жира ведущей ноги — **коррелят NIRS-аттенюации** |
| `muscle_to_fat_leg` | `dominant_leg_lean_mass / dominant_leg_fat_mass` | Отношение мышц к жиру ведущей ноги |

`leg_fat_pct` — ключевой признак для интерпретации NIRS: чем больше жировая прослойка над VL, тем сильнее аттенюация сигнала Train.Red и тем «оптимистичнее» смотрит SmO₂.

---

## Импутация `dominant_leg_lean_mass` и `dominant_leg_fat_mass`

У 2 из 14 участников (S006, S010) эти поля отсутствуют в HDF5.

По остальным 12 участникам вычислены стабильные коэффициенты:

| Коэффициент | Значение | CV |
|---|---|---|
| `leg_lean / skeletal_muscle_mass` | **0.271** | 6% |
| `leg_fat / body_fat_mass` | **0.157** | 10% |

Импутация:
```python
leg_lean = skeletal_muscle_mass * 0.271  # если missing
leg_fat  = body_fat_mass        * 0.157  # если missing
```

Колонки `dominant_leg_lean_mass_imputed` и `dominant_leg_fat_mass_imputed` (bool) — флаг импутации. Используются вместе с `dominant_leg_lean_mass_is_missing`.

Обоснование: коэффициенты получены из той же измерительной системы (биоимпеданс), что и исходные данные → ошибка импутации ≈ 6–10%, что сопоставимо с погрешностью прибора.

---

## Связанные файлы

- `dataset_pipeline/subjects.py` — чтение HDF5 и вычисление производных
- `dataset/subjects.parquet` — выходная таблица
- `preprocessing/nirs_pipeline.md` — `leg_fat_pct` как ковариат NIRS
