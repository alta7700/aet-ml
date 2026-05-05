# Структура `finaltest.h5`

Ниже зафиксирована текущая структура файла `finaltest.h5`:
- корневые метаданные;
- каналы в `/channels`;
- единая структура каждого канала.

## Корневые метаданные

### Унаследованные от исходного HDF5
- `createdAt`
- `filenameTemplate`
- `profile`
- `recorderPluginId`
- `recordingStartSessionMs`
- `recordingStartWallMs`
- `sessionStartWallMs`
- `writer`

### Из `summary.csv`
- `subject_name`
- `height`
- `weight`
- `age`
- `sex`
- `body_fat_mass`
- `skeletal_muscle_mass`
- `dominant_leg_lean_mass`
- `dominant_leg_fat_mass`
- `phase_angle`
- `dominant_leg_circumference`
- `stop_time`
- `stop_time_sec`

### Метаданные LT2
- `lt2_method`
- `lt2_power_w`
- `lt2_lactate_mmol`
- `lt2_time_sec`
- `lt2_interval_start_sec`
- `lt2_interval_end_sec`
- `lt2_interval_start_power_w`
- `lt2_interval_end_power_w`
- `lt2_pchip_power_w`
- `lt2_pchip_lactate_mmol`
- `lt2_pchip_time_sec`
- `lt2_pchip_delta_power_w`
- `lt2_pchip_delta_time_sec`
- `lt2_hrvt2_time_sec`
- `lt2_hhb_breakpoint_time_sec`
- `lt2_hhb_peak_time_sec`
- `lt2_refined_time_sec`
- `lt2_refined_sources`

## Каналы в `/channels`

- `power.label`
- `trigno.vl.avanti`
- `trigno.rf.avanti`
- `trigno.vl.avanti.gyro.x`
- `trigno.vl.avanti.gyro.y`
- `trigno.vl.avanti.gyro.z`
- `trigno.rf.avanti.gyro.x`
- `trigno.rf.avanti.gyro.y`
- `trigno.rf.avanti.gyro.z`
- `moxy.smo2`
- `moxy.thb`
- `zephyr.rr`
- `zephyr.hr`
- `zephyr.dfa_a1`
- `train.red.smo2`
- `train.red.hbdiff`
- `train.red.smo2.unfiltered`
- `train.red.o2hb.unfiltered`
- `train.red.hhb.unfiltered`
- `train.red.thb.unfiltered`
- `train.red.hbdiff.unfiltered`
- `lactate`

## Структура каждого канала

Для каждого канала используется одинаковое дерево:

```text
/channels/<channel_name>/timestamps
/channels/<channel_name>/values
```

## Краткое пояснение по каналам

- `power.label` — ступени мощности велоэргометра
- `trigno.vl.avanti` — ЭМГ Vastus Lateralis
- `trigno.rf.avanti` — ЭМГ Rectus Femoris
- `trigno.*.gyro.*` — гироскопические оси датчиков Delsys
- `moxy.smo2` — SmO2 из исходного ANT+ канала
- `moxy.thb` — THb из исходного ANT+ канала
- `zephyr.rr` — RR-интервалы
- `zephyr.hr` — частота сердечных сокращений
- `zephyr.dfa_a1` — готовый канал DFA-a1 от логгера
- `train.red.smo2` — выровненный качественный SmO2 из Train.Red
- `train.red.hbdiff` — HbDiff из Train.Red
- `train.red.smo2.unfiltered` — несглаженный SmO2
- `train.red.o2hb.unfiltered` — несглаженный O2Hb
- `train.red.hhb.unfiltered` — несглаженный HHb
- `train.red.thb.unfiltered` — несглаженный THb
- `train.red.hbdiff.unfiltered` — несглаженный HbDiff
- `lactate` — дискретные точки лактата
