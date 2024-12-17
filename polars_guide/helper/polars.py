from datetime import datetime, timedelta
import random
from typing import Any
import numbers
from dataclasses import dataclass
from collections import ChainMap
from functools import cache
import polars as pl
from polars.exceptions import InvalidOperationError
from IPython.display import display_html, display_markdown, display_pretty
from helper.python import keydefaultdict


class DataCapturer:
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
    