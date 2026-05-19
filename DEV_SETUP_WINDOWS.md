# Установка проекта на Windows — для разработчика

Краткая инструкция: что скачать, что поставить, как запустить.

---

## 1. Скачать и установить (3 программы)

### Git for Windows
- Скачать: https://git-scm.com/download/win
- Установить со всеми **дефолтными** опциями (Next → Next → Install).
- Проверить: открой PowerShell, набери `git --version` — должна показаться версия.

### Python 3.11
- Скачать: https://www.python.org/downloads/release/python-3119/
  → **Windows installer (64-bit)**
- ❗ **На первом экране установщика обязательно поставь галку «Add python.exe to PATH»**, иначе ничего не заработает.
- Дальше "Install Now".
- Проверить: в PowerShell `python --version` → должно показать `Python 3.11.9`.

### Cursor (или VS Code)
- Скачать: https://cursor.com/
- Установить, войти в тот же аккаунт, что и на Ubuntu.

---

## 2. Развернуть проект

Открой **PowerShell** (нажми `Win`, набери `powershell`, Enter).

### 2.1. Клонировать с GitHub

```powershell
cd C:\
git clone https://github.com/Legenda2002/trading-search-app.git
cd trading-search-app
```

### 2.2. Принести данные (графики + индекс)

Скачай с Google Drive `trading-search-app-backup.zip` (2.5 ГБ) в `Загрузки`.

Распакуй ZIP **внутрь** `C:\trading-search-app\`:
- Правый клик на ZIP → «Извлечь всё...» → указать `C:\trading-search-app\` → «Извлечь».
- Если Windows спросит про замену файлов — **«Да для всех»** (код из ZIP идентичен коду из git, не страшно).

После распаковки в проекте должно быть:
```
C:\trading-search-app\
├── app\, scripts\, packaging\, .github\   ← код (из git)
├── samples\library\                       ← 17k графиков (из ZIP)
├── data\app.db                            ← база (из ZIP)
├── data\images\                           ← оригиналы + миниатюры (из ZIP)
├── data\index\descriptors\                ← ORB-дескрипторы (из ZIP)
├── data\index\embeddings.npz              ← DINOv2 эмбеддинги (из ZIP)
└── data\hf_cache\                         ← модель DINOv2 (из ZIP)
```

### 2.3. Создать виртуальное окружение и поставить зависимости

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt
```

Долгий шаг (~5 мин): pip скачает PyTorch CPU, PySide6, transformers (~1.5 ГБ).

**Если PowerShell ругнётся** на `Activate.ps1` («running scripts is disabled»):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Подтверди `Y`, потом повтори `.\venv\Scripts\Activate.ps1`.

### 2.4. Починить пути в БД (одна команда)

В БД пути от Ubuntu (`/home/maruf/...`) — на Windows они недействительны. Запусти миграцию:

```powershell
python -m scripts.migrate_paths
```

Должно вывести:
```
Detected old roots: ['/home/maruf/trading-search-app']
Rewriting to new root: C:\trading-search-app
Backup created: data\app.db.bak
Updated 17663/17663 rows
```

Бэкап старой БД создаётся автоматически как `data\app.db.bak`.

---

## 3. Запустить

```powershell
python -m app.main
```

Открывается окно приложения. Через **«Библиотека → Просмотр базы»** убедись, что 17 664 графика на месте — готов искать.

---

## 4. Дальше в работе

Меняешь код в Cursor → коммит → пуш:

```powershell
git add .
git commit -m "что изменил"
git push
```

Если хочешь сделать **новый Release с .exe** для клиента:
```powershell
git tag -a v0.2.0 -m "v0.2.0 — описание изменений"
git push origin v0.2.0
```

GitHub Actions сам соберёт `.exe` на серверах GitHub за ~10 минут и положит ZIP в Releases. На своей машине PyInstaller тебе не нужен.

---

## Частые грабли

| Проблема | Решение |
|---|---|
| `python` не найден | Не поставил галку "Add to PATH". Переустанови Python с галкой. |
| `Activate.ps1` запрещён | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `pip install` падает на `torch` | Проверь интернет; повтори с `--no-cache-dir`. |
| Программа не находит графики после старта | Запусти ещё раз `python -m scripts.migrate_paths`. |
| Эмбеддинги не находятся | `python -m scripts.build_embeddings` — пересоберёт за ~10 мин. |
| Хочу новый набор графиков добавить | Меню «Библиотека → Импортировать папку». |

---

## Если совсем по-новой (без архива)

Если по какой-то причине ZIP не работает / потерялся — можно собрать всё с нуля:

```powershell
git clone https://github.com/Legenda2002/trading-search-app.git
cd trading-search-app
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt

# кинуть свои PNG-графики в samples\library\ вручную,
# затем проиндексировать:
python -m scripts.index_folder samples\library
python -m scripts.build_embeddings

python -m app.main
```

Индексация 17k графиков ~10–30 мин (зависит от CPU). Эмбеддинги ещё ~10 мин.
