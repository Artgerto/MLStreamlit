import numpy as np
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from scipy.stats import ttest_ind
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import altair as alt

st.write("""
# Промежуточная аттестация
#### Задание выполнено Лысенко Оксаной
""")

uploaded_file = st.file_uploader("Выберите файл датасета", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Отображение первых данных датасета")
    st.dataframe(df.head())
    sel_columns = st.multiselect("Выберите две колонки для визуализации распределения:", df.columns, max_selections=2)
    if len(sel_columns) == 2:
        st.write("Визуализация распределения")
        for col in sel_columns:
            is_numeric = is_numeric_dtype(df[col].dtype)
            if is_numeric:
                chart = alt.Chart(df).mark_bar().encode(alt.X(col), y='count()')
            else:
                chart = alt.Chart(df).mark_arc().encode(theta='count()', color=col)
            st.altair_chart(chart, use_container_width=True)
        test_algo = st.selectbox("Выберите алгоритм проверки гипотез:", ['t-тест', 'хи-квадрат'])
        if test_algo == 't-тест':
            if (is_numeric_dtype(df[sel_columns[0]].dtype) & is_string_dtype(df[sel_columns[1]].dtype)) \
                    | (is_string_dtype(df[sel_columns[0]].dtype) & is_numeric_dtype(df[sel_columns[1]].dtype)):
                category_column = sel_columns[0] if is_string_dtype(df[sel_columns[0]].dtype) else sel_columns[1]
                values_column = sel_columns[0] if is_numeric_dtype(df[sel_columns[0]].dtype) else sel_columns[1]
                check_categories = st.multiselect("Выберите две категории для проверки:", df[category_column].unique(),
                                                  max_selections=2)
                if len(check_categories) == 2:
                    st.write("Данные выбранных колонок")
                    st.dataframe(df[sel_columns][(df[category_column] == check_categories[0])
                                                 | (df[category_column] == check_categories[1])].dropna())

                    # Проверка гипотезы
                    group1 = df[df[category_column] == check_categories[0]]
                    group2 = df[df[category_column] == check_categories[1]]
                    st.write("Результаты t-теста:")
                    st.write(ttest_ind(group1[values_column].dropna(), group2[values_column].dropna()))
                    st.write('Если p-значение меньше 0,05, то нулевая гипотеза t-критерия отклоняется')
            else:
                st.write(
                    "Для проверки гипотезы с помощью t-теста выберите один категориальный столбец, второй числовой")
        else:
            if is_string_dtype(df[sel_columns[0]].dtype) & is_string_dtype(df[sel_columns[1]].dtype):
                cdf = df[sel_columns].dropna()
                st.write("Данные выбранных колонок")
                st.dataframe(cdf)

                # Чтобы определить, являются ли результаты теста хи-квадрат статистически значимыми,
                # можно сравнить статистику теста с критическим значением хи-квадрат.
                # Если статистика теста больше критического значения хи-квадрат,
                # то результаты теста являются статистически значимыми.
                chi_q = 0.05  # обычно выбирают 0,01, 0,05 и 0,10
                # Кол-во степеней свободы
                chi_qf = (len(cdf[sel_columns[0]].unique()) - 1) * (len(cdf[sel_columns[1]].unique()) - 1)
                st.write(f"Кол-во степеней свободы: {chi_qf}")

                st.write("Определение критического значения хи-квадрата:")
                st.write(chi2.ppf(1 - chi_q, df=chi_qf))  # Поиск критического значения для 95%

                st.write("Подсчет значений по категориям")
                cross_tab = pd.crosstab(cdf[sel_columns[0]], cdf[sel_columns[1]], margins=True)
                cross_tab.index = np.append(sorted(cdf[sel_columns[0]].unique()), 'col_totals')
                cross_tab.columns = np.append(sorted(cdf[sel_columns[1]].unique()), 'row_totals')
                st.dataframe(cross_tab)

                col_count = len(cross_tab.columns) - 1
                row_count = len(cross_tab.index) - 1

                observed = cross_tab.iloc[0:row_count, 0:col_count]

                st.write("Рассчитанные ожидаемые частоты:")
                expected = np.outer(cross_tab["row_totals"][0:row_count],
                                    cross_tab.loc["col_totals"][0:col_count]) / cross_tab["row_totals"][row_count]
                expected = pd.DataFrame(expected)
                expected.index = sorted(cdf[sel_columns[0]].unique())
                expected.columns = sorted(cdf[sel_columns[1]].unique())
                st.dataframe(expected)

                chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
                st.write(f"Рассчитанная хи-статистика: {chi_squared_stat}")

                p_value = 1 - chi2.cdf(x=chi_squared_stat, df=chi_qf) # Поиск значения вероятности
                st.write("Рассчитанное значение вероятности:")
                st.write(p_value)

                st.write("###### Выполнение хи-квадрат теста")
                chi_sq, chi_p_value, chi_df, chi_expected = chi2_contingency(observed=observed)
                st.write(f"Хи-статистика: {chi_sq}")
                st.write(f"Значение вероятности: {chi_p_value}")
                st.write(f"Кол-во степеней свободы: {chi_df}")
                st.write(f"Ожидаемые частоты, основанные на предельных суммах таблицы:")
                st.dataframe(chi_expected)

                st.write("Тест следует использовать только в том случае, если наблюдаемая и ожидаемая частоты "
                         "в каждой ячейке составляют не менее 5. "
                         "Если значение вероятности высокое, то связи между двумя переменными практически нет.")
            else:
                st.write(
                    "Для проверки гипотезы с помощью t-теста выберите два категориальных столбца")
