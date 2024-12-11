import polars as pl
from polars.exceptions import InvalidOperationError


def list_eval(df: pl.DataFrame, *exprs, **named_exprs):
    names = set()
    for expr in exprs:
        names.update(expr.meta.root_names())
    for expr in named_exprs.values():
        names.update(expr.meta.root_names())

    row_nr = "_row_nr_"

    is_lazy = isinstance(df, pl.LazyFrame)

    res = (
        df.lazy()
        .with_row_count(name=row_nr)
        .explode(list(names))
        .group_by(row_nr, maintain_order=True)
        .agg(*exprs, **named_exprs)
        .drop(row_nr)
    )

    if is_lazy:
        return res
    else:
        return res.collect()


pl.DataFrame.list_eval = list_eval
pl.LazyFrame.list_eval = list_eval


def try_cast_to_number(s, int_type=pl.Int64, float_type=pl.Float64):
    try:
        return s.str.strip_chars().cast(int_type)
    except InvalidOperationError:
        try:
            return s.str.strip_chars().cast(float_type)
        except InvalidOperationError:
            return s