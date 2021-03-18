import pandas as pd
import numpy as np

import sys
import os
from dataclasses import dataclass
from typing import Any



filename = sys.argv[1]
nans = []

df = pd.read_csv(filename, dtype='object', keep_default_na=True, na_values=nans)
#print("Loading file, using additional nan values: " + str(nans))

@dataclass
class Test:
    name: str
    condition: Any
    to_html: Any


def length_table(column):
    a = column.str.len().value_counts(normalize=True, dropna=True).round(4) * 100
    a.index.name = 'Value Lengths'
    a.name = '% of Total'
    a = a.to_frame()
    a['Examples'] = ""

    for i in range(0, len(a)):
        mask = column.str.len() == a.index[i]
        if column.loc[mask].nunique() > 3:
            sample_size = 3
        else:
            sample_size = column.loc[mask].nunique()
        example_values = np.random.choice(column.loc[mask].unique(), sample_size, replace=False)
        a['Examples'].values[i] = example_values

    a = a.to_markdown(tablefmt='html')
    return a


def unique_table(column):
    a = column.sort_values().value_counts(normalize=True, dropna=True).round(4) * 100
    a.index.name = 'Unique Values'
    a.name = '% of Total'
    a = a.to_markdown(tablefmt='html')
    return a

def sample_without_replacement(column):
    if column.dropna().nunique() > 3:
        sample_size = 3
    else:
        sample_size = column.dropna().nunique()
    example_values = np.random.choice(column.dropna().unique(), sample_size, replace=False)
    return example_values


tests = [
    Test(
        'all_nan',
        lambda column: column.isnull().all(),
        lambda column: "NAN values only"
    ),
    Test(
        'not_all_nan',
        lambda column: not column.isnull().all(),
        lambda column: "sample values, without replacement: " + str(sample_without_replacement(column))
    ),
    Test(
        'has_nan',
        lambda column: column.hasnans,
        lambda column: "% NAN: " + str((column.isnull().sum().round(2) * 100 / len(column)).round(2)) + ", NAN format(s): " + str(column[column.isna()].unique())
    ),
    Test(
        'no_nan',
        lambda column: not column.hasnans,
        lambda column: "% NAN: no NAN values found"
    ),
    Test(
        'all_cells_same_value',
        lambda column: len(column.value_counts(dropna=False)) == 1,
        lambda column: "all cells have same value: " + str(column[0])
    ),
    Test(
        'all_unique',
        lambda column: max(column.value_counts(dropna=False)) == 1,
        lambda column: "all non-NAN values are unique"
    ),
    Test(
        'unique_id',
        lambda column: column.nunique() == len(column),
        lambda column: "column can function as a unique identifier, first value is of form: " + str(column[0])
    ),
    Test(
        'row_index_0',
        lambda column: all(column.astype(int, errors='ignore').sort_values().eq(column.index.values)),
        lambda column: "column values form a row index, initialized at zero"
    ),
    Test(
        'row_index_1',
        lambda column: all(column.astype(int, errors='ignore').sort_values().eq(column.index.values + 1)),
        lambda column: "column values form a row index, initialized at one"
    ),
    Test(
        'strip_and_casefold_collapse',
        lambda column: column.str.strip().str.casefold().nunique == column.nunique(),
        lambda column: "whitespace and/or casefolding merges one or more values"
    ),
    Test(
        'salesforceid_15',
        lambda column: column.str.fullmatch("[a-zA-Z0-9]{15}").any(),
        lambda column: "one of more entries could be Salesforce IDs (15 characters)"
    ),
    Test(
        'salesforceid_18',
        lambda column: column.str.fullmatch("[a-zA-Z0-9]{18}").any(),
        lambda column: "one of more entries could be Salesforce IDs (18 characters)"
    ),
    Test(
        'numeric_only',
        lambda column: column.dropna().str.isnumeric().all(),
        lambda column: "all values are numeric only (or nan)"
    ),
    Test(
        'numeric_only_unique_over_max',
        lambda column: column.dropna().str.isnumeric().all() & column.nunique() > 25,
        lambda column: "min value: " + str(column.min(skipna=True)) + "  max value: " + str(column.max(skipna=True))
    ),
    Test(
        'alpha_only',
        lambda column: column.dropna().str.isalpha().all(),
        lambda column: "all values are alpha only (or nan) (no spaces, no specials)"
    ),
    Test(
        'common_lengths',
        lambda column: 0 < column.str.len().nunique() < 5,
        lambda column: length_table(column)
    ),
    Test(
        'unique_under_max',
        lambda column: 0 < column.nunique() <= 25,
        lambda column: unique_table(column)
    )

]


def run_tests(column):
    tests_passing = []
    for test in tests:
        if test.condition(column):
            tests_passing.append(test)
    return tests_passing


results = df.apply(run_tests, axis=0)


def build_output(column_name, column, column_results):
    column_results_list = [result.name for result in column_results]

    html = ""
    html += """<h1><b>""" + str(column_name) + """</b></h1>"""
    html += """Triggered Tests: """ + str(column_results_list) + """<br><br>"""
    html += "Unique Values, count: " + str(column.nunique()) + ", as share of non-NAN entries: " + str(round(column.nunique() * 100 / len(column.notnull()), 4)) + "%"
    html += """<br><br>"""

    for test in column_results:
        html += test.to_html(column)
        html += """<br><br>"""
    return html


def run_page(results):
    filled_page = ""
    a = results.map(lambda result: [key.name for key in result])
    for i in range(0, len(results)):
        column_name = results.keys()[i]
        column_results = results[i]
        column = df[column_name]

        filled_page += build_output(column_name, column, column_results)
    return filled_page


f = open("html_template.html", "r")
html_open = f.read()

page_output = run_page(results)

html_close = "</body></html>"

html_out = html_open + page_output + html_close

f = open("flenser_output.html", "w")
f.write(html_out)
f.close()

os.system("xdg-open flenser_output.html")
