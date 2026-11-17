# reqrel – виявлення семантичних дублікатів, уточнень і суперечностей у вимогах

Програмна реалізація курсової роботи «Метод виявлення семантичних дублікатів і суперечностей
у вимогах до інформаційних систем на основі трансформерних моделей» (КНУ імені Тараса Шевченка,
ОНП «Технології штучного інтелекту», 2026).

## Метод

1. **Етап 1 – відбір кандидатів.** Гібридна подібність (BM25 + косинус BGE-small, опукла комбінація, α = 0,5);
   для кожної вимоги – k найближчих сусідів (k = 3), |C| ≤ k·n.
2. **Етап 2 – двонапрямлене NLI.** Cross-encoder DeBERTaV3 (SNLI + MultiNLI) в обох напрямках:
   `s_dup = min(P_E(a,b), P_E(b,a))`, `s_con = max(P_C(a,b), P_C(b,a))`;
   взаємне слідування → DUP, однонапрямлене → REF, суперечність → CON, інакше NEU.
3. **Етап 3 – перевірка обмежень.** Числа/одиниці, модальність, заперечення (правила, табл. 2.1).

## Структура

```
reqrel/            пакет: data, retrieval, nli, constraints, pipeline, cache
experiments/       build_testset.py, exp1_retrieval.py … exp6_cn.py (експерименти E1–E6)
app.py, examples.json   вебінтерфейс (Streamlit) і 10 готових наборів вимог
data/train, test   корпуси UAV, WorldVista, PURE, OpenCOSS, IBM-UAV (оброблена версія, див. нижче)
data/pts.csv       тестовий набір PTS (240 пар, 4 класи)
data/requirements_example.txt, .csv   приклади файлів для завантаження в інтерфейс
data/paraphrase_testset_seeds.py   вимоги-зразки та їх варіанти
results/           CSV з результатами, cache/ – кешовані оцінки NLI
```

## Відтворення

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python experiments/build_testset.py      # формує data/pts.csv
python experiments/exp1_retrieval.py     # E1: Recall@k, частка пар, AP
python experiments/exp2_corpora.py       # E2: B1–B3, M1–M3, повний метод на корпусах
python experiments/exp3_pts.py           # E3: 4-класова оцінка на PTS, абляції, бутстреп
python experiments/exp4_transfer.py      # E4: перенесення керованих моделей на PTS
python experiments/exp5_timing.py        # E5: швидкодія
python experiments/exp6_cn.py            # E6: набір CN
```

Значення генераторів випадкових чисел – 42. Обчислення виконувалися на CPU без графічного прискорювача;
повний прогін займає близько 1,5 години, з кешем `results/cache` – кілька хвилин.

Використання як бібліотеки:

```python
from reqrel.pipeline import detect
reqs = ["The UAV shall fully charge in less than 3 hours.",
        "Charging the drone to full capacity is allowed to take up to 5 hours.",
        "The UAV shall be known as the Hummingbird."]
for r in detect(reqs, k=2):   # рекомендована конфігурація: гібрид, DeBERTaV3-base, числове правило
    print(reqs[r.i], "|", reqs[r.j], "->", r.label, round(r.s_con, 3))
```

## Інтерфейс

```bash
streamlit run app.py
```

Одна сторінка: вибір готового набору вимог (`examples.json`, 10 наборів із п'яти корпусів), введення власних вимог
або завантаження TXT/CSV; кнопка «Аналізувати» показує знайдені дублікати, уточнення та суперечності з оцінками
і дозволяє завантажити звіт CSV. Перший запуск завантажує моделі (≈1 ГБ).

Готові набори формує `build_examples.py`.

## Дані та ліцензії

Корпуси взято з відкритого репозиторію
https://github.com/Gechuyan/Enhancing-Requirement-Conflict-Detection-with-Data-Augmentation-and-Selective-Knowledge-Distillation
(файл dataset.zip); первинні джерела корпусів описано в роботі S3CDA (arXiv:2206.13690).
Моделі: BAAI/bge-small-en-v1.5, cross-encoder/nli-deberta-v3-{xsmall,small,base,large} (Hugging Face).
