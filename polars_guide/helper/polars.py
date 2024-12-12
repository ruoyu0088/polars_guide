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


def load_100knocks_data():
    pl.Config.set_fmt_str_lengths(100)
    
    dtypes = {
        'customer_id': str,
        'gender_cd': str,
        'postal_cd': str,
        'application_store_cd': str,
        'status_cd': str,
        'category_major_cd': str,
        'category_medium_cd': str,
        'category_small_cd': str,
        'product_cd': str,
        'store_cd': str,
        'prefecture_cd': str,
        'tel_no': str,
        'postal_cd': str,
        'street': str
    }
    
    df_customer = pl.read_csv("data/customer.csv", schema_overrides=dtypes)
    df_category = pl.read_csv("data/category.csv", schema_overrides=dtypes)
    df_product = pl.read_csv("data/product.csv", schema_overrides=dtypes)
    df_receipt = pl.read_csv("data/receipt.csv", schema_overrides=dtypes)
    df_store = pl.read_csv("data/store.csv", schema_overrides=dtypes)
    df_geocode = pl.read_csv("data/geocode.csv", schema_overrides=dtypes)

    return df_customer, df_category, df_product, df_receipt, df_store, df_geocode