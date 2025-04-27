import pandas as pd
import numpy as np
import html
import re

import platform
import sys
import os
from dataclasses import dataclass
from typing import Any


filename = sys.argv[1]
extra_nans = sys.argv[2:]

df = pd.read_csv(filename, dtype='object', keep_default_na=False)  # do not parse any NANs yet

nans = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
nans = nans + extra_nans

if not extra_nans:
    print("Using standard nan values: " + str(nans))
if extra_nans:
    print("Using standard and user-specified nan values: " + str(nans))

found_nans = df[df.isin(nans)].values
found_nans = np.unique(found_nans[~(pd.isnull(found_nans))])

nan_locations_all = df.apply(lambda column: column[column.isin(found_nans)].unique(), axis = 0)
nan_locations = nan_locations_all[nan_locations_all.apply(len).gt(0)]
nan_locations.index.name = 'column name'
nan_locations.name = 'NANs found'
html_nan_locations = nan_locations.to_markdown(tablefmt='html')

# Helper function to find values matching nans with surrounding spaces
def find_nans_with_spaces(column, nans_list):
    # Create a regex pattern: optional spaces + any nan value + optional spaces
    # Escape regex special characters in nan values first
    escaped_nans_for_regex = [re.escape(nan) for nan in nans_list if nan] # Exclude empty string from direct regex matching if needed
    # Handle empty string separately if it's in nans
    patterns = []
    if '' in nans_list:
        patterns.append(r'^\s+$') # Match strings containing only spaces

    if escaped_nans_for_regex:
         patterns.append(r'^\s*(' + '|'.join(escaped_nans_for_regex) + r')\s*$')

    if not patterns:
        # Return an empty array consistent with .unique() output
        return np.array([], dtype=object)

    full_pattern = '|'.join(patterns)
    # Operate on the column as strings
    col_str = column.astype(str)
    # Find values that match the pattern but are NOT exact matches in the original nans list
    matches = column[col_str.str.fullmatch(full_pattern, na=False) & ~column.isin(nans_list)]
    return matches.unique()

# Pre-calculate the results for the new test BEFORE replacing NaNs
nans_with_spaces_results = df.apply(lambda col: find_nans_with_spaces(col, nans))

# Replace identified NaNs with pandas NA object AFTER checking for spaced ones
df.replace(nans, pd.NA, inplace=True)


@dataclass
class Test:
    name: str
    condition: Any
    to_html: Any

#helper functions for Tests

def length_table(column):
    a = column.astype(str).str.len().value_counts(normalize=True, dropna=True).round(4) * 100
    a.index.name = 'Value Lengths'
    a.name = '% of Total'
    a = a.to_frame()
    a['Examples'] = ""

    for i in range(0, len(a)):
        mask = column.astype(str).str.len() == a.index[i]
        if column.loc[mask].nunique() > 3:
            sample_size = 3
        else:
            sample_size = column.loc[mask].nunique()
        example_values = np.random.choice(column.loc[mask].unique(), sample_size, replace=False)
        # Convert numpy array to string for tabulate
        a['Examples'].values[i] = str(example_values)

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
        lambda column: "% NAN: " + str((column.isnull().sum().round(2) * 100 / len(column)).round(2))
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
        lambda column: pd.to_numeric(column, errors='coerce').notna().all() and all(pd.to_numeric(column).sort_values().eq(column.index.values)),
        lambda column: "column values form a row index, initialized at zero"
    ),
    Test(
        'row_index_1',
        lambda column: pd.to_numeric(column, errors='coerce').notna().all() and all(pd.to_numeric(column).sort_values().eq(column.index.values + 1)),
        lambda column: "column values form a row index, initialized at one"
    ),
    Test(
        'strip_and_casefold_collapse',
        lambda column: column.astype(str).str.strip().str.casefold().nunique() < column.nunique(),
        lambda column: "whitespace and/or casefolding merges one or more values"
    ),
    Test(
        'salesforceid_15',
        # Use a more specific regex requiring both letters and numbers
        lambda column: column.astype(str).str.fullmatch(r"^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]{15}$").any(),
        lambda column: f"one or more entries match the structure of Salesforce IDs (15 chars, mixed alpha/numeric): '{column.dropna().unique()[0]}'"
    ),
    Test(
        'salesforceid_18',
        # Use a more specific regex requiring both letters and numbers
        lambda column: column.astype(str).str.fullmatch(r"^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]{18}$").any(),
        lambda column: f"one or more entries match the structure of Salesforce IDs (18 chars, mixed alpha/numeric): '{column.dropna().unique()[0]}"
    ),
    Test(
        'numeric_only',
        lambda column: column.dropna().astype(str).str.isnumeric().all(),
        lambda column: "all values are numeric only (or nan)"
    ),
    Test(
        'numeric_only_unique_over_max',
        lambda column: column.dropna().astype(str).str.isnumeric().all() & column.nunique() > 25,
        lambda column: "min value: " + str(pd.to_numeric(column, errors='coerce').min(skipna=True)) + "  max value: " + str(pd.to_numeric(column, errors='coerce').max(skipna=True))
    ),
    Test(
        'alpha_only',
        lambda column: column.dropna().astype(str).str.isalpha().all(),
        lambda column: "all values are alpha only (or nan) (no spaces, no specials)"
    ),
    Test(
        'matches_nan_with_spaces',
        lambda column: len(nans_with_spaces_results[column.name]) > 0,
        lambda column: f"Values matching NANs with surrounding spaces found: {html.escape(str(nans_with_spaces_results[column.name]))}"
    ),
    Test(
        'leading_trailing_spaces',
        # Check if any non-NA string value has different length after stripping
        lambda column: column.dropna().astype(str).pipe(lambda s: s.str.len().ne(s.str.strip().str.len())).any(),
        lambda column:(
            lambda cleaned_strings:
                f"Column contains values with leading/trailing spaces: {str(sample_without_replacement(cleaned_strings[cleaned_strings.ne(cleaned_strings.str.strip())]))}"
                )(column.dropna().astype(str))
    ),
    Test(
        'boolean_like',
        lambda column: column.nunique() == 2,
        lambda column: f"column appears boolean (exactly two distinct non-NA values): '{column.dropna().unique()[0]}' and '{column.dropna().unique()[1]}'"
    ),
Test(
        'leading_zeros',
        lambda column: column.dropna().astype(str).str.match(r'^0[0-9]+$').any(),
        lambda column: (
            lambda preprocessed_strings:
                f"Contains leading zeros: {str(sample_without_replacement(preprocessed_strings[preprocessed_strings.str.match(r'^0[0-9]+$')]
                ).tolist())}"
                )(column.dropna().astype(str))
),
    Test(
        'common_lengths',
        lambda column: 0 < column.astype(str).str.len().nunique() < 5,
        lambda column: length_table(column.astype(str))
    ),
    Test(
        'unique_under_max',
        lambda column: 0 < column.nunique() <= 25,
        lambda column: unique_table(column)
    ),
    Test(
        'contains_numeric',
        lambda column: column.astype(str).str.isnumeric().any(),
        lambda column: pd.to_numeric(column, errors='coerce').describe().to_markdown(tablefmt='html')
    )

]


def run_tests(column):
    tests_passing = []
    for test in tests:
        if test.condition(column):
            tests_passing.append(test)
    return tests_passing


results = df.apply(run_tests, axis=0)


top_table = pd.DataFrame(index=range(len(results)), columns=['Column Name', 'NANs found', 'Tests Triggered'])
for i in range(0, len(results)):
    top_table.values[i][0] = results.keys()[i]
    # Use .iloc for positional access to prevent FutureWarning
    top_table.values[i][1] = str(nan_locations_all.iloc[i])
    # Use .iloc for positional access to prevent FutureWarning
    top_table.values[i][2] = [result.name for result in results.iloc[i]]

html_top_table = top_table.to_markdown(tablefmt='html')


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
    for i in range(0, len(results)):
        column_name = results.keys()[i]
        # Use .iloc for positional access to prevent FutureWarning
        column_results = results.iloc[i]
        column = df[column_name]

        filled_page += build_output(column_name, column, column_results)
    return filled_page


f = open("html_template.html", "r")
html_open = f.read()

html_filename = str(filename) + """<br><br>"""
html_row_col = "Rows: " + str(df.shape[0]) + ", Columns: " + str(df.shape[1]) + """<br><br>"""
# Escape the lists *only* when creating the HTML string
html_nan = "NANs searched for: " + str([html.escape(nan) for nan in nans]) + """<br>""" + "NANs found: " + str([html.escape(nan) for nan in found_nans]) + """<br><br>"""

page_output = run_page(results)

html_close = "</body></html>"

html_out = html_open + html_filename + html_row_col + html_nan + html_top_table + page_output + html_close

f = open("flenser_output.html", "w")
f.write(html_out)
f.close()

print("Results saved as 'flenser_output.html'")

if platform.system() == "Linux":
    print("Attempting to open results in your default browser")
    os.system("xdg-open flenser_output.html")
