<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-dark.svg">
    <img alt="JackknifeTools" src="docs/src/assets/logo.svg" width="520">
  </picture>
</p>

[![CI](https://github.com/akio-tomiya/JackknifeTools.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/JackknifeTools.jl/actions/workflows/CI.yml)

**Author**: A. Tomiya

[English manual](README.md)

JackknifeTools は、Monte Carlo のデータに Jackknife と block Jackknife で誤差を
付けるための小さな Julia パッケージです。プライマリな物理量、非線形な
セカンダリな物理量、相関のある複数の測定列から作る量、ヒストグラムについて、
中心値と誤差を求めます。自己相関時間の見積もりと、block size の走査もできます。

依存は Julia 標準ライブラリの `Statistics` だけです。メモリはサンプル数 N に
比例する分しか使わず、平均とヒストグラムは O(N) の時間で計算するので、
長い Monte Carlo の履歴もそのまま渡せます。

バージョン 1.0 より前は JkTools という名前でした。関数名 (`jk_meanerror`、
`jk_hist` など) は変わっていません。

## インストール

```julia
pkg> add JackknifeTools
```

General レジストリへの登録が終わるまでは、GitHub から入れてください。

```julia
pkg> add https://github.com/akio-tomiya/JackknifeTools.jl
```

## 使い方の例

```julia
using JackknifeTools
using Statistics

data = randn(10_000)
A = 3 .+ data .+ 0.1 .* randn(10_000)   # 同じサンプルで測った 2 つの列
B = 2 .+ data .+ 0.1 .* randn(10_000)

jk_meanerror(data)                                  # 平均と Jackknife 誤差
jk_meanerror(data; block=50)                        # block Jackknife
jk_meanerror(data, x -> var(x, corrected=false))    # 非線形な量
jk_meanerror(data, "binder"; block=50)              # 定義済みのキー
jk_meanerror(A, B; block=50) do a, b                # 相関のある複数の列
    mean(a) / mean(b)
end
```

どの関数も `(中心値, 誤差)` を返します。中心値は全サンプルで計算した値で、
誤差は Jackknife の replica から計算します。

## 機能

| 用途 | 関数 |
| --- | --- |
| 平均と誤差 | `jk_meanerror(data)`、`jk_meanerror(data, func)`、`jk_meanerror(data, key)` |
| 相関のある複数の列 | `jk_meanerror(func, A, B, ...)` と do ブロックの形 |
| ベクトル値の量と共分散 | 配列を返す `func` を渡した `jk_meanerror`、`jk_meancov` |
| Jackknife の replica | `jk_replicas`、`jk_error`、`jk_cov` |
| 誤差棒つきヒストグラム | `jk_hist(samples; bins=20)`、`jk_histogram(samples, edges)` |
| 自己相関 | `jk_autocorr`、`jk_tau_int`、`jk_block_scan` |
| 添字の組 | `jk_index`、`jk_index_set`、`jk_block_index`、`jk_block_index_set` |
| 以前からの block 用 API | `jk_block_meanerror`、`jk_block_histogram`、`jk_block_hist` |
| ヘルプ (英語) | `?jk_meanerror`、`?jk_hist`、`jk_help()` |

どの関数も `block` キーワードを受け付けます。指定しなければ、1 点抜きの
通常の Jackknife になります。

使えるキー (`String` でも `Symbol` でもよい) は次のとおりです。

```julia
"mean", "average"          # mean(x)
"sus", "susceptibility"    # var(x, corrected=false)
"binder", "bin"            # Binder ratio: mean(x .^ 4) / mean(x .^ 2)^2
```

## データの形

- ベクトルは、1 要素が 1 サンプル (1 配位) です。
- 行列は、1 行が 1 サンプルです。たとえば相関関数 `C[config, t]` です。
  `jk_meanerror(C)` は列ごとの平均と誤差を返します。
- ベクトルのベクトルも使えます。内側のベクトルが 1 サンプル分です。
- 複素数のデータでは、誤差は `complex(実部の誤差, 虚部の誤差)` になります。

`func` には同じ形のサンプルが渡されます。`func` は数、複素数、配列のどれを
返してもかまいません。配列なら成分ごとに誤差を返します。

## Block Jackknife

Block Jackknife は、連続した block を 1 つずつ抜きます。Markov chain
Monte Carlo のように自己相関があるデータに使います。

```julia
jk_meanerror(data; block=50)
jk_meanerror(data, "sus"; block=50)
```

`length(data)` が block size で割り切れないときは、先頭の余りを捨ててから
同じ大きさの block に分けます。中心値も replica も、捨てた後のデータで
計算します。

```julia
jk_block_index_set(1:5, 2)
# [[4, 5], [2, 3]]
```

この例では `1` を捨て、残りを `[2, 3]` と `[4, 5]` に分けています。
block は少なくとも 2 つ必要です。

### block size の選び方

```julia
tau = jk_tau_int(data)          # Sokal の自動 window、c = 6
tau.tau_int, tau.error, tau.window

scan = jk_block_scan(data; blocks=1:200)
scan.block_sizes, scan.nblocks, scan.values, scan.errors
```

平均の block Jackknife 誤差は、block size とともに増え、block size が
`tau_int` より十分大きくなると、通常の誤差の `sqrt(2 tau_int)` 倍あたりで
平坦になります。その平坦な領域で、block 数が十分残る block size を選びます。

## 相関のある複数の列

同じ配位で測った複数の量から作る量は、全ての列を 1 回の呼び出しに渡します。
どの replica でも全ての列から同じサンプル (または同じ block) を抜くので、
列どうしの相関が誤差に入ります。

```julia
ratio, ratio_err = jk_meanerror((a, b) -> mean(a) / mean(b), A, B; block=50)
```

`mean(A)` と `mean(B)` の誤差を別々に求め、独立とみなして合成すると、
大きく間違えることがあります。チュートリアルには、誤差を 10 倍以上
大きく見積もってしまう例があります。

## ベクトル値の量、共分散、replica

```julia
# 相関関数 C[config, t] から有効質量
meff(c) = (ct = vec(mean(c; dims=1)); log.(ct[1:end-1] ./ ct[2:end]))

m, m_err = jk_meanerror(C, meff; block=4)
m, m_cov = jk_meancov(C, meff; block=4)        # 相関のある fit 用の共分散

rep = jk_replicas(C, meff; block=4)            # Jackknife サンプルごとの値
jk_error(rep.replicas), jk_cov(rep.replicas)
```

## 誤差棒つきヒストグラム

`samples` の 1 要素が 1 サンプルです。数でもベクトルでもよく、たとえば
1 つのゲージ配位で測った Dirac 固有値の列を入れます。

```julia
eigenvalues_by_config = [
    [0.002, 0.011, 0.018],
    [0.004, 0.010, 0.026],
    [0.001, 0.016, 0.021],
]

hist = jk_hist(eigenvalues_by_config; bins=collect(0.0:0.01:0.03))
hist.edges, hist.centers, hist.values, hist.errors
```

`bins` は整数でも、bin の端の列でもかまいません。bin は `[edge[i], edge[i+1])`
で、最後の右端だけは含めます。範囲外の値、`NaN`、`±Inf` は数えません。
`density=true` で bin 幅で割り、それ以外の規格化は `scale` で掛けます。

```julia
volume = 32^3 * 8
hist = jk_hist(eigenvalues_by_config; bins=20, block=2, scale=1 / volume, density=true)
```

JackknifeTools は描画パッケージに依存しません。Plots.jl なら次のように描けます。

```julia
using Plots
bar(hist.centers, hist.values; yerror=hist.errors, bar_width=hist.edges[2] - hist.edges[1], label=false)
```

`examples/histogram_errorbars.jl` も見てください。

## チュートリアル

[`tutorials/README.md`](tutorials/README.md) は、熱浴法で生成した 2 次元 Ising
模型のデータを使ったチュートリアルです。統計誤差、誤差の伝播、通常の
Jackknife と block Jackknife、自己相関関数と自己相関時間、block size の選び方、
相関のある複数の列から作る量、ヒストグラムを、実行結果と図つきで説明して
います。

```bash
julia --project=. tutorials/ising2d_jackknife.jl
julia --project=. tutorials/make_tutorial_artifacts.jl
```

## 入力の検査

通常の Jackknife にはサンプルが 2 つ以上必要です。Block Jackknife には正の
整数の block size と、先頭の余りを捨てた後に 2 つ以上の block が必要です。
複数の列は同じ数のサンプルを持つ必要があります。条件を満たさない入力には
`ArgumentError` を投げます。

## ライセンス

MIT License です。
