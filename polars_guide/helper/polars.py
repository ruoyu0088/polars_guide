from io import StringIO
import json
from datetime import datetime, timedelta
import random
from typing import Any
import numbers
from dataclasses import dataclass
from collections import ChainMap
from functools import cache
import pyarrow as pa
import numpy as np
import polars as pl
from polars.exceptions import InvalidOperationError
from IPython.display import display_html, display_markdown, display_pretty
from helper.python import keydefaultdict

class DataCapturer:
    """
    A utility class for capturing and logging DataFrames during method chaining in data pipelines.

    This class allows tracking intermediate results in a data pipeline by logging 
    DataFrames associated with method names dynamically. It enables debugging and 
    analyzing the state of data at various stages of the pipeline.

    Example
    -------

    ```python
    df = pl.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    cap = DataCapturer()

    (
    df
    .with_columns(
        C=pl.col('A') % 2
    )
    .pipe(cap.before_group)
    .group_by('C')
    .agg(
        pl.col('B').max()
    )
    .sort('B')
    )

    print(cap.before_group)
    ```
    """

    def __init__(self):
        self._current_name = None  # Tracks the name of the current method being logged
        self._logs = {}            # Stores logged data by method name

    def __getattr__(self, name):
        # Return logged data if the name exists in the logs, otherwise set the current name for logging
        if name in self._logs:
            return self._logs[name]
        else:
            self._current_name = name
            return self

    def __call__(self, df):
        # Log the DataFrame under the current method name and return it for method chaining
        self._logs[self._current_name] = df
        return df
    

class ExprCapturer:
    def __init__(self):
        self._current_name = None
        self._logs = {}            

    def __getattr__(self, name):
        if name in self._logs:
            return self._logs[name]
        else:
            self._current_name = name
            if self._current_name not in self._logs:
                self._logs[self._current_name] = []
            return self

    def _log_data(self, data):
        self._logs[self._current_name].append(data)
        return data

    def __call__(self, expr):
        return expr.map_batches(self._log_data)


markdown_css = """
code {
  color: darkgreen;
}
"""


@dataclass
class MethodCall:
    method: str
    args: tuple
    kw: dict
    input: Any
    output: Any

    def to_panel(self):
        import panel as pn
        def make_pane(data):
            try:
                html = data._repr_html_()
                return pn.pane.HTML(html)
            except AttributeError:
                return pn.pane.HTML(f"<pre>{data}</pre>")

        args = "\n".join([f"* `{str(arg)}`" for arg in self.args])
        kws = "\n".join([f"* {name}=`{str(value)}`" for name, value in self.kw.items()])

        method = f"""
## {self.method}

### args
{args}

### kws
{kws}
        """.strip()
        args_pane = pn.pane.Markdown(method, stylesheets=[markdown_css])
        input_pane = make_pane(self.input)
        output_pane = make_pane(self.output)

        return pn.Column(args_pane, pn.Row(input_pane, output_pane))


class PipeLogger:
    def __init__(self, obj):
        self.__current_obj = obj
        self.__history = [obj]

    def __getattr__(self, name):
        if name.startswith("_ipython") or name.startswith("_repr"):
            raise AttributeError()
        obj = getattr(self.__current_obj, name)
        self.__history.append(obj)
        self.__current_obj = obj
        return self

    def __call__(self, *args, **kw):
        import types

        obj = self.__current_obj(*args, **kw)
        self.__history.append(obj)
        self.__current_obj = obj

        input, method, output = self.__history[-3:]

        if isinstance(input, MethodCall):
            input = input.output

        del self.__history[-2:]
        if isinstance(method, types.MethodType):
            method_name = method.__func__.__name__
        else:
            method_name = method.__name__

        mc = MethodCall(method=method_name, input=input, output=output, args=args, kw=kw)
        self.__history.append(mc)
        return self

    def __getitem__(self, name):
        obj = self.__current_obj[name]
        self.__history.append(obj)
        self.__current_obj = obj
        return self

    def _repr_mimebundle_(self, **kwargs):
        if not hasattr(self, '_tab'):
            self._tab = self.create_tab()
        return self._tab._repr_mimebundle_(**kwargs)

    def create_tab(self):
        import panel as pn
        children = []
        titles = []
        for obj in self.__history:
            if isinstance(obj, MethodCall):
                children.append(obj.to_panel())
                titles.append(obj.method)
        return pn.Tabs(*zip(titles, children))


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

def get_caller_line(level=2):
    import inspect
    frame = inspect.currentframe()
    for i in range(level):
        frame = frame.f_back
    caller_source = inspect.getsourcelines(frame)[0]
    current_lineno = frame.f_lineno  # Current line number

    target_line = caller_source[current_lineno - 1]
    return target_line

def load_data(folder_path, dtypes, level=2):
    target_line = get_caller_line(level=level)
    
    # Extract variable names before the '='
    var_names = target_line.split('=')[0].strip()
    var_names = [name.strip() for name in var_names.split(',')]

    # Load CSV files matching the variable names
    results = []
    for var_name in var_names:
        if var_name.startswith('df_'):
            var_name = var_name[3:]
        file_path = f"{folder_path}/{var_name}.csv"
        try:
            df = pl.read_csv(file_path, schema_overrides=dtypes)
            results.append(df)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found for variable '{var_name}': {file_path}")

    if len(results) == 1:
        return results[0]
    else:
        return results


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
    return load_data('data', dtypes, level=3)            


def get_expr_functions(root, show_error=False):
    env = {}
    
    # Get all attributes in the `polars` module
    all_attrs = dir(root)
    
    # Filter to get only functions
    functions = [
        attr for attr in all_attrs
        if not attr.startswith("_") and callable(getattr(root, attr))
    ]

    for func_name in functions:
        func = getattr(root, func_name)

        try:
            if func.__annotations__['return'] == 'Expr': 
                env[func_name] = func
        except KeyError:
            if show_error:
                print(f"KeyError {func}")
        except AttributeError:
            if show_error:
                print(f"AttributeError {func}")
        except TypeError:
            if show_error:
                print(f"TypeError {func}")
            
    return env

@cache
def get_env():
    env = get_expr_functions(pl)
    env2 = get_expr_functions(pl.Expr)
    for key, val in env2.items():
        if key in env:
            key = f'{key}_'
        env[key] = val

    namespaces = [
        ('list', pl.expr.expr.ExprListNameSpace),
        ('str', pl.expr.expr.ExprStringNameSpace),
        ('bin', pl.expr.expr.ExprBinaryNameSpace),
        ('arr', pl.expr.expr.ExprArrayNameSpace),
        ('struct', pl.expr.expr.ExprStructNameSpace),
        ('cat', pl.expr.expr.ExprCatNameSpace),
        ('dt', pl.expr.expr.ExprDateTimeNameSpace),
        ('name', pl.expr.expr.ExprNameNameSpace),
        ('meta', pl.expr.expr.ExprMetaNameSpace),
    ]
        
    for prefix, namespace in namespaces:
        env2 = get_expr_functions(namespace)
        for key, val in env2.items():
            env[f'{prefix}_{key}'] = val
    
    env['col'] = pl.col
    return env


def polars_exprs(func):
    import builtins
    def inner_func(**kw):
        env = {}
        env.update(builtins.__dict__)
        env.update(func.__globals__)
        env.update(get_env())
        for name in func.__code__.co_names:
            if name.startswith("c_"):
                env[name] = pl.col(name[2:])
            
        for key, val in kw.items():
            if isinstance(val, str):
                val = pl.col(val)
            env[key] = val

        return eval(func.__code__, env)
    return inner_func


def with_columns_chain(self: pl.DataFrame, *args, **kw) -> pl.DataFrame:
    if kw:
        df = self.lazy()
        for key, expr in kw.items():
            df = df.with_columns(expr.alias(key))
        return df.collect()
    else:
        df = self.lazy()
        for expr in args:
            df = df.with_columns(expr)
        return df.collect()


if getattr(pl.DataFrame, "with_columns_chain", None) is not with_columns_chain:
    setattr(pl.DataFrame, "with_columns_chain", with_columns_chain)   

def create_datetime_sample_data(n=10):
    now = datetime(2024, 12, 10, 10, 32, 14)

    start_datetimes = []
    for _ in range(n):
        start_datetimes.append(now)
        now += timedelta(seconds=int(random.uniform(1, 20) * 3600))
    
    end_datetimes = [start + timedelta(seconds=random.randint(20, 1000)) for start in start_datetimes]
    categories = [random.choice(['A', 'B', 'C', 'D']) for _ in range(n)]
    
    df = pl.DataFrame({
        "start": start_datetimes,
        "end": end_datetimes,
        "category": categories
    })
    return df
    

class batch_base:
    def __init__(self, func, kw=None):
        """
        Initialize the batch processing class with the function and optional keyword arguments.
        """
        self._func = func
        self._kw = kw if kw is not None else {}

    def __call__(self, args=None, **kw):
        """
        Execute the batch processing function with arguments converted to a specific format (e.g., Arrow or NumPy).
        """
        # If keyword arguments are passed, return a new instance with updated kw
        if kw:
            return self.__class__(self._func, kw)
        
        # Convert series in args using from_series method of the class
        if args is not None:
            args = [self.from_series(s) for s in args]
        
        # Call the function with the arguments and the stored keyword arguments
        result = self._func(*args, **self._kw)

        # Convert result to a series if it's of the expected array type
        if isinstance(result, self.array_type):
            return self.to_series(result)
        return result

class series_batch(batch_base):
    array_type = type(None)
    from_series = staticmethod(lambda s:s)
    to_series = None

class pyarrow_batch(batch_base):
    """
    Batch processor for pyarrow arrays. Converts Series to Arrow arrays and back.
    """
    array_type = pa.Array
    from_series = staticmethod(pl.Series.to_arrow)
    to_series = staticmethod(pl.from_arrow)


class numpy_batch(batch_base):
    """
    Batch processor for numpy arrays. Converts Series to NumPy arrays and back.
    """
    array_type = np.ndarray
    from_series = staticmethod(pl.Series.to_numpy)
    to_series = staticmethod(pl.Series)


def when_map(col_name, *args, default_value=None):
    if isinstance(col_name, str):
        col = pl.col(col_name)
    else:
        col = col_name

    expr = pl
    for from_values, to_value in zip(args[::2], args[1::2]):
        expr = expr.when(col.is_in(from_values)).then(to_value)
    return expr.otherwise(default_value)


def match(*exprs):
    "helper function for creating pl.when().then().when().then().otherwise() quickly"
    inv_exprs = list(exprs[::-1])
    res = pl
    while inv_exprs:
        e1 = inv_exprs.pop()
        try:
            e2 = inv_exprs.pop()
        except IndexError:
            res = res.otherwise(e1)
            break

        res = res.when(e1).then(e2)
    return res


def agg(df, items):
    """
    Perform aggregation operations on specified columns of a Polars DataFrame and return the results.

    Args:
        df (pl.DataFrame): The input Polars DataFrame on which the aggregations will be performed.
        items (dict): A dictionary where keys are column names in `df`, and values are lists of aggregation method names (strings) 
                      to apply to the corresponding column.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the aggregated results. 
                      The rows correspond to the columns being aggregated, and the columns represent the aggregation names.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> items = {"a": ["mean", "max"], "b": ["min"]}
        >>> result = agg(df, items)
        >>> print(result)
        shape: (2, 4)
        ┌────────┬──────┬──────┬──────┐
        │ column ┆ mean ┆ max  ┆ min  │
        │ ---    ┆ ---  ┆ ---  ┆ ---  │
        │ str    ┆ f64  ┆ i64  ┆ i64  │
        ╞════════╪══════╪══════╪══════╡
        │ a      ┆ 2.0  ┆ 3    ┆ null │
        │ b      ┆ null ┆ null ┆ 4    │
        └────────┴──────┴──────┴──────┘        
    """    
    exprs = {
        key: pl.struct(**{agg_name: getattr(pl.col(key), agg_name)() for agg_name in agg_names})
        for key, agg_names in items.items()
    }
    return df.select(**exprs).transpose(include_header=True, column_names=["agg"]).unnest("agg")


def concat_with_keys(dfs, keys, key_column_name='key'):
    """
    Concatenate multiple Polars DataFrames vertically, adding a key column to identify the source of each row.

    Args:
        dfs (list of pl.DataFrame): A list of Polars DataFrames to concatenate.
        keys (list): A list of keys corresponding to each DataFrame in `dfs`.
        key_column_name (str): The name of the column to store the keys. Defaults to 'key'.

    Returns:
        pl.DataFrame: A single concatenated DataFrame with an additional key column.

    Example:
        >>> import polars as pl
        >>> df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})
        >>> keys = ["df1", "df2"]
        >>> result = concat_with_keys([df1, df2], keys)
        >>> print(result)
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ key ┆ a   ┆ b   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ df1 ┆ 1   ┆ 3   │
        │ df1 ┆ 2   ┆ 4   │
        │ df2 ┆ 5   ┆ 7   │
        │ df2 ┆ 6   ┆ 8   │
        └─────┴─────┴─────┘        
    """    
    dfs = [df.with_columns(pl.lit(key).alias(key_column_name)) 
           for df, key in zip(dfs, keys)]
    return pl.concat(dfs, how='vertical').select(key_column_name, pl.exclude(key_column_name))


def pivot_with_margins(df, on, index, expr):
    """
    Create a pivot table from a Polars DataFrame, including marginal totals.

    Parameters:
    ----------
    df : pl.DataFrame
        The input Polars DataFrame.
    on : str
        The column name to use as the pivot column.
    index : str
        The column name to use as the index for the pivot table.
    expr : pl.Expr
        The Polars expression to aggregate values (e.g., a sum or count operation).

    Returns:
    -------
    pl.DataFrame
        A pivot table with marginal totals added for both rows and columns, as well as a grand total.

    Notes:
    -----
    - The marginal totals are calculated by aggregating across all levels of `on` and/or `index`.
    - The grand total is calculated by aggregating across all rows and columns.

    Example:
    -------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "Category": ["A", "A", "B", "B"],
    ...     "Subcategory": ["X", "Y", "X", "Y"],
    ...     "Values": [10, 20, 30, 40]
    ... })
    >>> expr = pl.col("Values").sum()
    >>> result = pivot_with_margins(df, on="Subcategory", index="Category", expr=expr)
    >>> print(result)

    shape: (3, 4)
    ┌──────────┬─────┬─────┬─────┐
    │ Category ┆ X   ┆ Y   ┆ All │
    │ ---      ┆ --- ┆ --- ┆ --- │
    │ str      ┆ i64 ┆ i64 ┆ i64 │
    ╞══════════╪═════╪═════╪═════╡
    │ A        ┆ 10  ┆ 20  ┆ 30  │
    │ B        ┆ 30  ┆ 40  ┆ 70  │
    │ All      ┆ 40  ┆ 60  ┆ 100 │
    └──────────┴─────┴─────┴─────┘    

    This will create a pivot table with totals for each category, subcategory, and a grand total.
    """    
    all_on = pl.lit('All').alias(on)
    all_index = pl.lit('All').alias(index)

    return (
        pl.concat([
            df.group_by(index, on).agg(expr),
            df.group_by(index).agg(all_on, expr),
            df.group_by(on).agg(all_index, expr),
            df.select(all_on, all_index, expr),
        ], how='diagonal')
         .pivot(on=on, index=index)
    )


def update_with_mask(value_expr, cond_expr, values_expr):
    """
    Updates a column or expression with new values based on a condition mask.
    
    This function applies a condition (`cond_expr`) to determine where updates
    should be made. For every `True` in the condition mask, corresponding values 
    from `values_expr` are used to replace values in `value_expr`. The updates 
    are applied sequentially.

    Parameters:
    ----------
    value_expr : pl.Expr
        The Polars expression or column to be updated.
        
    cond_expr : pl.Expr
        A Polars expression that evaluates to a boolean mask, where `True`
        indicates positions to update.
        
    values_expr : list, tuple, or pl.Expr
        The values to use for updating. Can be a list or tuple for sequential updates
        or a Polars expression.

    Returns:
    -------
    pl.Expr
        A new Polars expression where `value_expr` is updated with values
        from `values_expr` based on the condition mask.

    Notes:
    ------
    - The condition mask (`cond_expr`) determines the positions to update. Only
      `True` positions are considered.
    - If `values_expr` is a list or tuple, it is automatically converted to a 
      Polars literal (`pl.lit`).
    - Updates are applied sequentially, starting from the first `True` in the 
      condition mask.

    Example:
    --------
    ```python
    import polars as pl

    df = pl.DataFrame({
        "value": [1, 2, 3, 4, 5],
    })

    cond = pl.col("value") % 2 == 0
    updates = [10, 20]

    updated_df = df.with_columns(
        update_with_mask(pl.col("value"), cond, updates).alias("updated_value")
    )
    print(updated_df)
    ```
    """    
    index = pl.when(cond_expr).then(1).cum_sum() - 1
    if isinstance(values_expr, (tuple, list)):
        values_expr = pl.lit(pl.Series(values_expr))
    return pl.coalesce(values_expr.get(index), value_expr)


def to_dataframe(data):
    import re
    from io import StringIO
    data = re.sub(r'[ \t]+', ',', data.strip())
    return pl.read_csv(StringIO(data))    


def align_op(df1, df2, op, on='index', how='left', fill_value=0):
    """
    Align two Polars DataFrames on a specified key column, then perform an operation on common columns.

    Args:
        df1 (pl.DataFrame): The base DataFrame to align and modify.
        df2 (pl.DataFrame): The secondary DataFrame whose values will be aligned and used for operations.
        op (callable): A Polars expression representing the operation to perform (e.g., pl.Expr.add, pl.Expr.sub).
        on (str): The name of the column to use as the key for alignment. Defaults to 'index'.
        fill_value (int or float): The value to fill in for missing entries after the join. Defaults to 0.

    Returns:
        pl.DataFrame: A new DataFrame with the same schema as `df1`, but with the operation applied to common columns.

    Example:
        >>> import polars as pl
        >>> df1 = pl.DataFrame({
        ...     "index": ["a", "b", "c"],
        ...     "X": [1, 2, 3],
        ...     "Y": [4, 5, 6]
        ... })
        >>> df2 = pl.DataFrame({
        ...     "index": ["a", "c"],
        ...     "X": [10, 20],
        ...     "Y": [30, 40]
        ... })
        >>> align_op(df1, df2, pl.Expr.sub)
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ index │ X   │ Y   │
        ├───────┼─────┼─────┤
        │ a     │ -9  │ -26 │
        │ b     │ 2   │ 5   │
        │ c     │ -17 │ -34 │
        └───────┴─────┴─────┘

    Notes:
        - The `op` argument must be a callable that takes two Polars expressions and returns a Polars expression.
        - Only columns that are common between `df1` and `df2` (excluding the alignment column) are processed.
        - The resulting DataFrame will have the same structure and columns as `df1`.
    """
    # Find common columns between the two DataFrames, excluding the alignment column
    common_columns = list(set(df1.columns) & set(df2.columns))
    if not isinstance(on, list):
        on = [on]
    for on_col in on:
        if on_col in common_columns:
            common_columns.remove(on_col)

    # Perform the operation using lazy evaluation
    df_res = (
        df1.lazy()
        .join(df2.lazy(), on=on, how=how, coalesce=True)  # Align rows by the key column
        .pipe(lambda df:df.fill_null(fill_value) if fill_value is not None else df) # Fill missing values
        .with_columns(
            [
                op(pl.col(col), pl.col(f"{col}_right")).alias(col)  # Apply operation to common columns
                for col in common_columns
            ]
        )
        .select(df1.columns)  # Preserve the structure of the original DataFrame
        .collect()  # Execute the lazy computation
    )
    return df_res
    

def expression_replace(expr, mapper=None, **kw):          
    if mapper is None:
        mapper = kw

    def convert_column(expr):
        if 'Column' in expr:
            column_name = expr['Column']
            if column_name in mapper:
                # Replace the column name using the mapper
                to_replace = mapper[column_name]
                if isinstance(to_replace, str):
                    expr['Column'] = to_replace
                else:
                    expr = to_replace
        return expr
        
    for key, val in mapper.items():
        if isinstance(val, pl.Expr):
            mapper[key] = json.loads(val.meta.serialize(format='json'))
            
    expr_json = json.loads(expr.meta.serialize(format='json'), object_hook=convert_column)
    new_expr = pl.Expr.deserialize(StringIO(json.dumps(expr_json)), format='json')
    return new_expr

def sympy_to_polars(expr, symbol_map={}):
    import numbers
    from functools import reduce
    import operator    
    import sympy as sp
    
    functions_map = {
        sp.Abs: "abs",
        sp.acos: "arccos",
        sp.acosh: "arccosh",
        sp.asin: "arcsin",
        sp.asinh: "arcsinh", 
        sp.atan: "arctan",
        sp.atanh: "arctanh",
        sp.cos: "cos",
        sp.cosh: "cosh",
        sp.cot: "cot",
        sp.exp: "exp",
        sp.log: "log",
        sp.sign: "sign",
        sp.sin: "sin",
        sp.sinh: "sinh",
        sp.tan: "tan",
        sp.tanh: "tanh",
    }
    
    def convert(expr):
        match expr:
            case sp.Symbol(name=name):
                if expr in symbol_map:
                    res = symbol_map[expr]
                elif name in symbol_map:
                    res = symbol_map[name]
                else:
                    res = name

                if isinstance(res, str):
                    res = pl.col(res)
                elif isinstance(res, numbers.Real):
                    res = pl.lit(float(res))
                return res
                
            case sp.Number() | sp.NumberSymbol():
                return pl.lit(float(expr))
                
            case sp.Add(args=args):
                return reduce(operator.add, [convert(arg) for arg in args])
                
            case sp.Mul(args=args):
                return reduce(operator.mul, [convert(arg) for arg in args])
                
            case sp.Pow(args=[base, exp]):
                return convert(base) ** convert(exp)
                
            case sp.Function(args=[arg]):
                func = type(expr)
                if func in functions_map:
                    return getattr(convert(arg), functions_map[func])()
                else:
                    raise NotImplementedError(f"unknown function: {expr}")        
    
            case _:
                raise NotImplementedError(f"Unsupported operation: {expr}")
                
    return convert(expr)


def dtype_pretty_format(obj, level=0, indent_size=2):
    indent = ' ' * (indent_size * level)

    def is_primitive(dtype):
        return not isinstance(dtype, (pl.List, pl.Struct))

    if isinstance(obj, pl.Schema):
        lines = [f"{indent}Schema("]
        for name, dtype in obj.items():
            if is_primitive(dtype):
                lines.append(f"{' ' * (indent_size * (level + 1))}{name}: {dtype}")
            else:
                dtype_str = dtype_pretty_format(dtype, level + 2, indent_size)
                lines.append(f"{' ' * (indent_size * (level + 1))}{name}:\n{dtype_str}")
        lines.append(f"{indent})")
        return "\n".join(lines)

    elif isinstance(obj, pl.List):
        inner = obj.inner
        if is_primitive(inner):
            return f"{indent}List({inner})"
        inner_str = dtype_pretty_format(inner, level + 1, indent_size)
        return f"{indent}List(\n{inner_str}\n{indent})"

    elif isinstance(obj, pl.Struct):
        lines = [f"{indent}Struct("]
        for field in obj.fields:
            name, dtype = field.name, field.dtype
            if is_primitive(dtype):
                lines.append(f"{' ' * (indent_size * (level + 1))}{name}: {dtype}")
            else:
                dtype_str = dtype_pretty_format(dtype, level + 2, indent_size)
                lines.append(f"{' ' * (indent_size * (level + 1))}{name}:\n{dtype_str}")
        lines.append(f"{indent})")
        return "\n".join(lines)

    else:
        return f"{indent}{repr(obj)}"    