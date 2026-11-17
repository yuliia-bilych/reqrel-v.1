"""Демонстраційний інтерфейс методу виявлення дублікатів, уточнень і суперечностей у вимогах.
Запуск:  streamlit run app.py
"""
import io, json, os
import pandas as pd
import streamlit as st

from reqrel import retrieval, nli, constraints

ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = json.load(open(os.path.join(ROOT, "examples.json"), encoding="utf-8"))
MODEL = "cross-encoder/nli-deberta-v3-base"
K = 3
UA = {"duplicate": "Дублікат", "refinement": "Уточнення", "contradiction": "Суперечність", "neutral": "Немає відношення"}
ROW_COLOR = {"Суперечність": "#fde4d8", "Дублікат": "#dbe8fa", "Уточнення": "#d7f2e6"}
ORDER = {"Суперечність": 0, "Дублікат": 1, "Уточнення": 2, "Немає відношення": 3}

st.set_page_config(page_title="Дублікати та суперечності у вимогах", layout="wide")
st.title("Виявлення семантичних дублікатів і суперечностей у вимогах")
st.markdown("<style>.block-container{max-width:1100px}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def analyse(reqs: tuple) -> tuple:
    reqs = list(reqs)
    S = retrieval.similarity(reqs, "hybrid_cc")
    cand = sorted(retrieval.topk_pairs(S, min(K, len(reqs) - 1)))
    scores = nli.score_pairs([(reqs[i], reqs[j]) for i, j in cand], MODEL)
    rows = []
    for (i, j), s in zip(cand, scores):
        lab = constraints.fuse(nli.decide(s), constraints.check(reqs[i], reqs[j]), ("numeric",))
        rows.append({"№": f"{i + 1}–{j + 1}", "Вимога a": reqs[i], "Вимога b": reqs[j], "Тип відношення": UA[lab],
                     "Оцінка дублювання": s.s_dup, "Оцінка суперечності": s.s_con})
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Тип відношення", "Оцінка суперечності"],
                        key=lambda c: c.map(ORDER) if c.name == "Тип відношення" else -c).reset_index(drop=True)
    return df, len(cand)


# ---------- 1. Вимоги ----------
st.subheader("1. Вимоги")
source = st.radio("Джерело вимог", ["Готовий набір", "Власні вимоги", "З файлу"], horizontal=True,
                  label_visibility="collapsed")
reqs = []
if source == "Готовий набір":
    name = st.selectbox("Набір вимог", list(EXAMPLES))
    reqs = EXAMPLES[name]
    st.caption("Реальні вимоги з публічного корпусу та їх варіанти з тестового набору роботи.")
elif source == "Власні вимоги":
    txt = st.text_area("Кожна вимога з нового рядка (англійською мовою)", height=180)
    reqs = [r.strip() for r in txt.splitlines() if r.strip()]
else:
    up = st.file_uploader("TXT – одна вимога в рядку; CSV – вимоги в першому стовпці", type=["txt", "csv"])
    if up is not None:
        raw = up.read().decode("utf-8", errors="ignore")
        if up.name.lower().endswith(".csv"):
            reqs = pd.read_csv(io.StringIO(raw)).iloc[:, 0].dropna().astype(str).str.strip().tolist()
        else:
            reqs = [r.strip() for r in raw.splitlines() if r.strip()]

if reqs:
    st.markdown("\n".join(f"{n}. {r}" for n, r in enumerate(reqs, 1)))

with st.expander("Як працює метод"):
    st.markdown("**Етап 1.** Для кожної вимоги відбираються 3 найбільш схожі вимоги – лише ці пари перевіряються далі.  \n"
                "**Етап 2.** Модель логічного висновування перевіряє кожну пару в обох напрямках: "
                "чи випливає одна вимога з іншої і чи суперечить їй.  \n"
                "**Додатково** перевіряються числові обмеження з однаковими одиницями вимірювання.")

# ---------- 2. Результати ----------
if st.button("Аналізувати", type="primary", disabled=len(reqs) < 2):
    with st.spinner("Аналіз… (перший запуск завантажує моделі, це може зайняти кілька хвилин)"):
        df, n_cand = analyse(tuple(reqs))
    st.subheader("2. Результати")
    found = df[df["Тип відношення"] != "Немає відношення"]
    n = len(reqs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Вимог", n)
    c2.metric("Усіх пар", n * (n - 1) // 2)
    c3.metric("Відібрано для перевірки", n_cand)
    c4.metric("Знайдено відношень", len(found))
    if len(found):
        view = found.copy()
        for col in ("Оцінка дублювання", "Оцінка суперечності"):
            view[col] = view[col].map(lambda v: f"{v:.2f}".replace(".", ","))
        view = view.set_index("№")
        try:  # row colour by relation type (pandas Styler needs jinja2 >= 3.1.5)
            view = view.style.apply(lambda r: [f"background-color: {ROW_COLOR.get(r['Тип відношення'], '')}"] * len(r), axis=1)
        except (AttributeError, ImportError):
            pass
        st.table(view)
    else:
        st.info("Дублікатів, уточнень і суперечностей не знайдено.")
    st.caption("**№** – номери вимог зі списку. **Оцінка дублювання** – імовірність того, що кожна з двох вимог "
               "випливає з іншої (0–1). **Оцінка суперечності** – імовірність того, що вимоги не можуть виконуватися "
               "одночасно (0–1). **Уточнення** – одна вимога випливає з іншої лише в одному напрямку, тобто є конкретнішою.")
    st.download_button("Завантажити звіт (CSV)", df.round(3).to_csv(index=False).encode("utf-8-sig"), "report.csv", "text/csv")
