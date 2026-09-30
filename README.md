# draft-prefix-budget-allocator

运行于 Linux CPU 的批次草稿前缀选择引擎（Go 标准库实现，无模型下载、无外部服务调用）。

## 问题

在固定的核验位置预算内，为一批请求分配从起点开始的连续草稿前缀，使本轮预期接受词数最大。

- 每个请求有标识、有序候选词，以及每个位置在此前词全部接受条件下的条件接受概率。
- 位置 `i` 的收益是前缀连乘 `p[0]*p[1]*...*p[i]`，不是单位置概率；总收益是所有选中位置收益之和。
- 每个请求只能选零个词，或从位置 0 开始的完整前缀，禁止越过未选择位置。
- 每个位置耗费一个预算单位：候选充足时用满预算；候选不足时全部选择，并在结果的 `Remaining` 中报告剩余额度。
- 相同收益按「请求标识，再按位置」确定稳定顺序，输入请求顺序变化不改变结果。

## 算法

每个请求的边际收益序列（前缀连乘）非递增（每次乘以 `<= 1` 的概率），故每请求的总收益是选中长度的离散凹函数。用最小堆维护每条链当前的最前端位置，反复取最大边际收益即可得到全局最优分配；只有选中位置 `i` 后位置 `i+1` 才进入候选，前缀闭合由结构强制保证。完整贪心顺序只构建一次，可复用于同一输入的多个预算（`SelectMany`）。

长前缀的极小概率即使在 float64 中下溢为 0，也不影响前缀闭合与排序确定性。

## 用法

```go
e, err := draftselect.Build([]draftselect.Request{
    {ID: "a", Words: []string{"a0", "a1"}, Probs: []float64{1, 0.5}},
    {ID: "b", Words: []string{"b0"}, Probs: []float64{0.75}},
})
if err != nil { /* 重复标识 / 长度不符 / 概率越界或非有限 */ }

res, err := e.Select(2)
// res.Used, res.Remaining, res.Payoff
// res.Selections[i]: RequestID / Length / Positions / Words / Payoff

results, err := e.SelectMany([]int{0, 1, 2, 5})
```

错误哨兵：`ErrDuplicateID`、`ErrNegativeBudget`、`ErrLengthMismatch`、`ErrInvalidProbability`（覆盖越界与 NaN/Inf）。

## 测试

`go test ./...` 包含一个仅用于测试的小规模穷举参考（`math/big.Rat` 精确求和，独立枚举所有前缀分配），用于验证非均匀概率下的最优收益、前缀闭合、预算守恒，并覆盖稳定并列、候选不足、空批次/零预算、中途零概率、长前缀下溢、输入顺序无关与全部非法输入路径。
