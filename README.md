# VLM Defect Detection

Пайплайн визуального контроля качества на vision-language модели: **дообучение Qwen2.5-VL-3B через QLoRA** и сравнение трёх подходов к инспекции дефектов на одном эталонном наборе.

## Что это и для чего

Задача: по фотографии устройства определить **поломка/норма**, тип дефекта, локализацию и критичность — в структурированном JSON.

Сравниваются три слоя на одном eval-сете (419 изображений, 6 категорий MVTec AD):

| Слой | Подход |
|---|---|
| Baseline (CV) | ResNet18-фичи + logistic regression |
| VLM zero-shot | Qwen2.5-VL-3B «как есть» |
| VLM fine-tuned | QLoRA (r=16, 504 записи, чекпоинт-селекция) |

## Результаты (полный eval, 419 изображений)

| Подход | Accuracy | Precision | Recall | F1 | JSON | defect_type |
|---|---|---|---|---|---|---|
| Baseline (CV) | 0.733 | 0.909 | 0.640 | 0.751 | — | — |
| **VLM fine-tuned** | **0.778** | 0.768 | **0.928** | **0.840** | **1.00** | **0.47** |

Ключевое:
- **JSON validity 100%** — 419/419 ответов строго структурированы (structured output через JSON schema)
- **Recall 0.93** — модель находит 93% дефектов; CV-baseline пропускает 36%
- Профили осмысленно противоположны: CV точнее на норме, VLM надёжнее на браке. Для QC-контура «VLM-фильтр + человек» профиль VLM предпочтительнее
- **Масштабирование данных работает**: defect_type accuracy выросла 0.27 → 0.47 при росте датасета 178 → 504 записей

## История разработки

Проект прошёл 5 итераций; каждая проблема — с root-cause анализом, не наугад:

1. **Unsloth GGUF-экспорт теряет текстовый LoRA** (v2→v3). `save_pretrained_gguf` для Qwen2.5-VL молча экспортирует базовую текстовую модель без вшитых адаптеров. Доказано sha256-сравнением двух прогонов с разными лоссами — файлы байт-в-байт идентичны. Fix: явный `merge_and_unload()` через PEFT + конвертация llama.cpp.
2. **Смещение датасета** (v4): 89 дефектов и 0 нормальных примеров научили модель «всегда находить поломку» — 74 ложных срабатывания на good. Fix: сбалансированный датасет (504 записи, 252/252).
3. **Рассинхрон промптов** (v3): обучение шло с одним system-промптом, инференс с другим — модель вне распределения. Fix: single source of truth для промптов (импорт из `vlm.py` в генератор датасета).
4. **CPU-сборка llama.cpp под видом GPU** (v5): unsloth бандлит `linux-x64-cpu` бинарник, `-ngl` молча игнорируется — 150 c/изображение. Fix: официальный CUDA-релиз (150× ускорение eval).
5. **Переполнение диска чекпоинтами** (v5): optimizer state × 10 эпох = 30+ GB. Fix: `save_only_model=True` + чекпоинт-селекция по val.

Финальная пара: текст с LoRA (PEFT merge) + обученный vision-mmproj (unsloth-экспорт) — веса одного прогона, целостность проверена хэшами.

## Стек

- **Обучение**: Unsloth, QLoRA (4-bit, r=16), TRL SFTTrainer, Kaggle T4
- **Инференс**: Ollama (structured output), llama-server для масштабного eval
- **Baseline**: torchvision ResNet18 + scikit-learn
- **Данные**: MVTec AD (6 категорий, официальный зеркальный загрузчик)
- **Качество**: 35 offline-тестов, ruff, GitHub Actions CI

## Структура

```
notebooks/qlora_kaggle_v5.ipynb   # полный цикл: обучение → селекция → merge → GGUF → GPU-eval
scripts/build_eval_set.py          # эталонный сет + supervised-сплит (seed=42)
scripts/split_finetune_val.py      # стратифицированный train/val для селекции чекпоинтов
scripts/pack_manifest_zip.py       # упаковка датасета для Kaggle
scripts/train_baseline.py          # слой 1
scripts/local_full_eval.py         # полный eval через Ollama (resumable, overnight-safe)
src/defectbench/                   # датасет, метрики, baseline, VLM-клиент, FastAPI
results/                           # финальные отчёты обоих слоёв
```

## Воспроизведение

```powershell
python scripts\download_mvtec.py --categories bottle screw cable metal_nut transistor capsule --data-dir data
python scripts\build_eval_set.py --categories bottle screw cable metal_nut transistor capsule
python scripts\split_finetune_val.py
python scripts\pack_manifest_zip.py
python scripts\train_baseline.py            # слой 1, локально (CPU ~20 мин)
python scripts\local_full_eval.py           # слой 3 через Ollama (~6.5 ч, resumable)
# слой 2-3 обучение: notebooks\qlora_kaggle_v5.ipynb на Kaggle T4 (~2.5 ч)
```

## Ограничения

- supervised baseline обучается на половине дефектных тест-примеров (это часть протокола сравнения)
- location/severity — эвристическая разметка (MVTec не содержит аннотации критичности)
- defect_type accuracy ограничена ~20–40 примерами на класс; путь роста — данные
