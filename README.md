# QMorph

**QMorph** — генеративная NLP-модель на русском языке с морфологической декомпозицией и квантово-вдохновлёнными эмбеддингами.

## установка

`python == 3.9.*`

```bash
pip install -r requirements.txt
```

## обучение

обучить только qmorph:

```bash
./scripts/train_qmorph.sh config/config_books.yaml
```

обучить baseline + qmorph:

```bash
./scripts/train_all_models.sh config/config_books.yaml
```

## инференс

инференс только qmorph:

```bash
./scripts/infer_qmorph.sh "Я посмотрел на это создание,"
```

инференс baseline + qmorph:

```bash
./scripts/infer_all_models.sh "Я посмотрел на это создание,"
```

## сравнение моделей

скрипт считает метрики на тесте для baseline и qmorph, пишет отчёт в `reports/model_comparison.json`:

```bash
./scripts/compare_models.sh config/config_books.yaml
```

критерий победителя — меньшая perplexity по задаче next-lemma.

## тесты

```bash
python -m unittest discover -s tests -v
```
