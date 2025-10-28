# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import io
from typing import Any, Iterable

import pandas as pd

from superset.utils.core import GenericDataType


def _value_to_string(value: Any) -> str:
    """
    Convert a cell value to a safe string representation while preserving empty cells.
    """
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        # Non-numeric iterables (e.g. lists/tuples) raise on isna; treat them as non-null.
        pass

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")

    return str(value)


def _header_to_string(header: Any) -> str:
    """
    Create a readable header label, flattening multi-index headers if necessary.
    """
    if isinstance(header, tuple):
        parts = [part for part in header if part not in (None, "")]
        header = " | ".join(map(str, parts)) if parts else ""
    elif header is None:
        header = ""
    return str(header)


def _resolve_index_labels(
    df: pd.DataFrame, index_label: Any, *, levels: int
) -> list[Any]:
    if index_label is None:
        labels = list(df.index.names or [])
    elif isinstance(index_label, Iterable) and not isinstance(index_label, (str, bytes)):
        labels = list(index_label)
    else:
        labels = [index_label]

    if len(labels) < levels:
        remaining = (df.index.names or []) if df.index.names else []
        labels.extend(remaining[len(labels) : levels])
    if len(labels) < levels:
        labels.extend([None] * (levels - len(labels)))

    return labels[:levels]


def _auto_adjust_column_widths(
    worksheet: Any,
    df: pd.DataFrame,
    *,
    index: bool,
    index_label: Any,
    header: Any,
    startcol: int,
) -> None:
    """
    Expand Excel columns to fit the widest content (including headers if present).
    """

    def max_content_length(series: pd.Series) -> int:
        return max((len(_value_to_string(value)) for value in series), default=0)

    include_header = header is True or isinstance(header, (list, tuple))
    header_overrides = list(header) if isinstance(header, (list, tuple)) else []

    current_col = startcol
    if index:
        labels = _resolve_index_labels(df, index_label, levels=df.index.nlevels)
        for level, label in enumerate(labels):
            series = pd.Series(df.index.get_level_values(level))
            header_len = len(_header_to_string(label)) if include_header else 0
            data_len = max_content_length(series)
            width = max(header_len, data_len)
            adjusted_width = min(width + 2, 80) if width else 8
            worksheet.set_column(current_col, current_col, adjusted_width)
            current_col += 1

    for idx, column in enumerate(df.columns):
        series = pd.Series(df[column])
        header_value = None
        if include_header:
            if idx < len(header_overrides):
                header_value = header_overrides[idx]
            else:
                header_value = column
        header_len = len(_header_to_string(header_value)) if include_header else 0
        data_len = max_content_length(series)
        width = max(header_len, data_len)
        adjusted_width = min(width + 2, 80) if width else 8
        worksheet.set_column(current_col, current_col, adjusted_width)
        current_col += 1


def quote_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure to quote any formulas for security reasons.
    """
    formula_prefixes = {"=", "+", "-", "@"}

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: (
                f"'{x}"
                if isinstance(x, str) and len(x) and x[0] in formula_prefixes
                else x
            )
        )

    return df


def df_to_excel(df: pd.DataFrame, **kwargs: Any) -> Any:
    output = io.BytesIO()

    # make sure formulas are quoted, to prevent malicious injections
    df = quote_formulas(df)

    # pylint: disable=abstract-class-instantiated
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, **kwargs)
        sheet_name = kwargs.get("sheet_name", "Sheet1")
        worksheet = writer.sheets.get(sheet_name)
        if worksheet:
            _auto_adjust_column_widths(
                worksheet,
                df,
                index=kwargs.get("index", True),
                index_label=kwargs.get("index_label"),
                header=kwargs.get("header", True),
                startcol=int(kwargs.get("startcol", 0) or 0),
            )

    return output.getvalue()


def apply_column_types(
    df: pd.DataFrame, column_types: list[GenericDataType]
) -> pd.DataFrame:
    """
    Applies the column types to the dataframe to prepare for an excel export

    :param df: The dataframe to apply the column types to
    :param column_types: The types of the columns
    :return: The dataframe with the column types applied
    """
    for column, column_type in zip(df.columns, column_types, strict=False):
        if column_type == GenericDataType.NUMERIC:
            try:
                df[column] = pd.to_numeric(df[column])
                # if the number is too large, convert it to a string
                # Excel does not support numbers larger than 10^15
                df[column] = df[column].apply(
                    lambda x: str(x)
                    if isinstance(x, (int, float)) and abs(x) > 10**15
                    else x
                )
            except ValueError:
                df[column] = df[column].astype(str)
        elif pd.api.types.is_datetime64tz_dtype(df[column]):
            # timezones are not supported
            df[column] = df[column].astype(str)
    return df
