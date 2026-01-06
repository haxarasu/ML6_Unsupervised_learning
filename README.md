# ML6_Unsupervised_learning

## Структура

- `src/background/` — обработка видео и выделение фона (SVD).
- `src/books/` — подготовка матрицы для обучения моделей и их сравнение.
- `src/face_expression/` — подготовка данных для PCA и интерпретация эмоций.
- `src/image_compression/` — сжатие изображений.
- `src/visualizations/` — снижение размерности и оценка итогового качества.
- `datasets/` — данные для обучения.

## Установка зависимостей

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
---
## Запуск

Запуск происходит путем взаимодействия с ячейками .ipynb файлов в подпапках `src/`.

- `src/background/video_svd_minimal.ipynb` — отделение статичных и движущихся объектов на видео.
- `src/face_expression/faces.ipynb` — анализ выражений лица и изменение эмоций на фотографиях.
- `src/visualizations/visualizations.ipynb` — сравнение методов понижения размерности.
- `src/image_compression/images.ipynb` — сжатие изображений.
- `src/books/books.ipynb` — сравнение моделей в задаче предсказания возраста пользователя на основе его вектора оценок книг.