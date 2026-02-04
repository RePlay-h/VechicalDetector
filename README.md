# VehicleDetector — детекция транспорта на изображениях/кадрах с дронов

Проект для детекции транспортных средств на аэрофото/дрон-кадрах.  
В проекте есть:
- **Кастомная модель** детектора (EfficientNet (pretrained) + PAN + anchor-free head в стиле FCOS)
- **YOLO-модели** (минимум 2: быстрая и более точная)
- **Backend + Frontend**: FastAPI + Streamlit (пользователь выбирает модель из списка)
- **MLOps**: DVC (пайплайн/данные), MLflow (эксперименты), Docker (окружение)
- **Тесты**: `pytest`

---

## 1) Чек-лист требований

- [x] Тюнинг модели (кастомная + YOLO)
- [x] Backend (sync, без очереди)
- [ ] Backend (async с очередью) — для максимального балла (опционально)
- [x] Frontend (Streamlit)
- [x] Выбор модели пользователем (>=2 моделей)
- [x] Размещение решения на GitHub
- [ ] Видео-презентация приложения (после готовности backend+frontend)
- [ ] Деплой на Streamlit Cloud/аналог — доп. баллы (опционально)

---

## 2) Датасеты

- VisDrone: https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset  


---

## 3) Установка окружения

### Вариант A: Conda (Windows/Linux)
```bash
conda create -n vehdet python=3.10 -y
conda activate vehdet
pip install -r ./requirements.txt

```

---

## 4) DVC: пайплайн

Инициализация (один раз):
```bash
dvc init
```

Запуск стадий:
```bash
dvc repro
```

---

## 5) MLflow

### 5.1 Запуск MLflow сервера локально
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 5.2 Логирование YOLO в MLflow
Ultralytics умеет логировать в MLflow (при установленном `mlflow`).  
Пример для Windows PowerShell:
```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

python -m ml.yolo_train --params params.yaml --cfg-key yolo_fast
```

---

## 6) Backend (FastAPI)

Backend умеет:
- вернуть список доступных моделей
- прогнать изображение и вернуть bbox’ы (JSON)
- вернуть PNG с отрисованными bbox’ами

### 6.1 Запуск backend локально
Из корня репозитория:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Эндпоинты:
- `GET /models`
- `POST /detect/image`
- `POST /detect/image_render`

---

## 7) Frontend (Streamlit)

UI позволяет:
- загрузить изображение
- выбрать модель (`custom_fcos`, `yolo_fast`, `yolo_new`)
- настроить `score_thr` и `iou_thr`
- увидеть результат и список детекций

### 7.1 Запуск frontend локально
```bash
streamlit run frontend/app.py
```

### 7.2 Адрес backend (через переменную окружения)
По умолчанию используется `http://localhost:8000`.  
Можно переопределить:

**PowerShell**
```powershell
$env:API_URL="http://localhost:8000"
streamlit run frontend/app.py
```

**cmd**
```bat
set API_URL=http://localhost:8000
streamlit run frontend/app.py
```

---

## 8) Тесты

```bash
pytest -q
```

---

## 9) Частые проблемы

### StreamlitSecretNotFoundError
Не используйте `st.secrets` без файла `frontend/.streamlit/secrets.toml`.  
Рекомендуется использовать переменную окружения `API_URL`.

### Torch not compiled with CUDA enabled
Установлен CPU-only PyTorch — установите PyTorch с CUDA через conda (см. раздел 3).

### YOLO: Permission denied на data/yolo
Проверьте права на папку/блокировки файлов на Windows.

---


