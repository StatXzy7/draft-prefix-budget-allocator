# draft-prefix-budget-allocator

运行于 Linux CPU 的批次草稿选择引擎（Go 标准库实现）。

在固定核验位置预算内，为一批请求分配从起点开始的连续草稿前缀，
使本轮预期接受词数最大。位置 k 的收益为条件接受概率的连乘积
`p1*p2*...*pk`，总收益为全部选中位置收益之和；每个请求只能选零个词
或完整前缀。

## 用法

```go
eng, err := draft.NewEngine([]draft.Request{
    {ID: "A", Words: wordsA, AcceptProb: probsA},
    {ID: "B", Words: wordsB, AcceptProb: probsB},
})
if err != nil { /* 重复标识 / 长度不符 / 非法概率 */ }

// 同一输入可反复查询多个预算；候选不足时全部选择并报告剩余额度。
res, err := eng.Select(8)
fmt.Println(res.Used, res.Remaining, res.TotalPayoff)
```

核心实现：`draft/engine.go`；小规模穷举参考仅用于测试：
`draft/engine_test.go` 中的 `bruteForce`。

运行测试：

```sh
go test ./...
```
