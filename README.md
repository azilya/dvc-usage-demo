# Версионирование данных

## Создать репозиторий
```bash
git init
dvc init
```

## Ручной контроль данных и моделей

### Добавить в репозиторий данные
```bash
dvc add data/reviews_upd/ dom_rf_model
git add .gitignore dom_rf_model.dvc data/.gitignore data/reviews_upd.dvc
git commit -m "Initial commit"
git tag -a "v1.0" -m "first version"
```

### Вернуться к предыдущему состоянию
```bash
git checkout v1.0
dvc checkout
```

## Автоматический контроль через stages работы с моделью

### Добавить этап

```bash
dvc stage add -n train -d data -d model/ -d train.py -d main.py -o model/config.json -o model/pytorch_model.bin -m metrics.json python train.py
```
`-n`: название

`-d`: зависимость; файл, изменение которого запускает этап заново

`-o`: вывод; файл, который отслеживается как результат этапа

`-m`: метрики; файл с метриками

### Запустить этап

```bash
dvc repro {stage}
```
Проверить, изменились ли какие-то из зависимостей и запустить команду, если да.

```
dvc exp run [-n name]
```
Запустить весь пайплайн эксперимента с отслеживанием метрик

## dvc install

```
dvc install
```
Связывание git- и dvc-команд для более удобной интеграции:
* `dvc checkout`  после  `git checkout`
* `dvc push`  перед  `git push`
* `dvc status`  перед  `git commit`
