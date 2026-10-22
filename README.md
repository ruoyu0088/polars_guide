# Polars Guide

この本はChatGPTの助けを借りて、[Jupyter Book](https://jupyterbook.org)で作成したPolarsのガイドブックです。

https://ruoyu0088.github.io/polars_guide

## 使い方（日本語）

両言語で共通の資産は `shared/`（`helper/`, `data/`, `_static/`）にあります。
各言語のディレクトリにある同名のディレクトリは、ビルド前にコピーされる生成物で、Git では管理していません。

`data/` の一部のファイル（`data/stock/`、`*.arrow`、生成された `*.csv` など）は notebook の実行時に作られるもので、
`shared/` には含まれず、同期で削除されることもありません。

### ノートブックを実行する

```bash
python sync_assets.py --apply --all
cd polars_guide
jupyter lab
```

中国語版を動かす場合は、`cd polars_guide` を `cd polars_guide_cn` に読み替えてください。

- Jupyter は必ず本のディレクトリで起動してください。notebook の `data/...` と `import helper` はカレントディレクトリ基準です。
- 実行に必要なパッケージ: `polars`, `numpy`, `matplotlib`, `pandas`, `pyarrow`, `fastexcel`, `xlsxwriter`
  （描画や地理データの章では `holoviews`, `hvplot`, `panel`, `geopandas`, `geoviews`, `cartopy`, `shapely` も使用）。
- `requirements.txt` はビルド用で、実行用ではありません。

### ビルド

```bash
python sync_assets.py --apply --all
jupyter-book build polars_guide
jupyter-book build polars_guide_cn
```

### 資産の変更

`shared/` だけを編集し、`python sync_assets.py --apply --all` を実行してください。
`python sync_assets.py --check` で同期ずれを確認できます。

## 使用说明（中文）

两种语言共用的资源放在 `shared/`（`helper/`、`data/`、`_static/`）。
各语言目录下的同名目录是构建前复制的生成物，不纳入 Git。

`data/` 里有一部分文件（`data/stock/`、`*.arrow`、运行生成的 `*.csv` 等）是执行 notebook 时产生的，
不在 `shared/` 里，同步时也不会被删除。

### 运行 notebook

```bash
python sync_assets.py --apply --all
cd polars_guide_cn
jupyter lab
```

日文版把 `cd polars_guide_cn` 换成 `cd polars_guide`。

- Jupyter 必须从书目录启动，notebook 里的 `data/...` 和 `import helper` 都按当前目录解析。
- 运行依赖：`polars`、`numpy`、`matplotlib`、`pandas`、`pyarrow`、`fastexcel`、`xlsxwriter`
  （绘图和地理数据章节还需要 `holoviews`、`hvplot`、`panel`、`geopandas`、`geoviews`、`cartopy`、`shapely`）。
- `requirements.txt` 只用于构建站点，不包含运行依赖。

### 构建

```bash
python sync_assets.py --apply --all
jupyter-book build polars_guide
jupyter-book build polars_guide_cn
```

### 修改资源

只修改 `shared/`，然后运行 `python sync_assets.py --apply --all`。
`python sync_assets.py --check` 可以检查是否同步。
