// Package draftselect 实现一个运行于 Linux CPU 的批次草稿前缀选择引擎。
//
// 它在固定的核验位置预算内，为多个请求分配从起点开始的连续草稿前缀，
// 使本轮预期接受词数最大。每个位置 i（0 基）的收益为从请求起点到该位置
// 的条件接受概率连乘 p[0]*p[1]*...*p[i]（在 i 之前的词全部被接受的
// 条件下，词 i 被接受的概率），总收益为全部选中位置收益之和。
//
// 每个请求只能选择零个词，或从位置 0 开始的完整前缀；禁止越过未选择
// 的位置。每个位置耗费一个预算单位。
//
// 本包仅使用 Go 标准库，不下载模型，也不调用任何外部服务。
package draftselect

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"sort"
)

// Request 描述一个请求：标识、有序候选词，以及每个位置在此前词全部
// 接受条件下的条件接受概率。Words 与 Probs 必须等长。
type Request struct {
	ID    string
	Words []string
	Probs []float64
}

// Selection 是单个请求在某次预算求解下的选择结果。
type Selection struct {
	// RequestID 为对应的请求标识。
	RequestID string
	// Length 为选中前缀的长度（选中位置为 0..Length-1）。
	Length int
	// Positions 为选中位置（0 基），始终为 {0, 1, ..., Length-1}；
	// Length 为 0 时为空。
	Positions []int
	// Words 为选中位置对应的候选词，顺序与 Positions 一致。
	Words []string
	// Payoff 为该请求选中位置的收益之和（前缀连乘之和）。
	Payoff float64
}

// Result 是单个预算的求解结果。
type Result struct {
	// Budget 为求解时给出的预算。
	Budget int
	// Used 为实际使用的预算（全部选中位置数量）。
	Used int
	// Remaining 为未使用的剩余额度（候选不足时大于 0）。
	Remaining int
	// Payoff 为全部选中位置的收益总和。
	Payoff float64
	// Selections 按请求标识升序排列，包含批次中的每个请求
	// （空候选或零长度的请求也会出现）。
	Selections []Selection
}

var (
	// ErrDuplicateID 在批次中出现重复请求标识时返回。
	ErrDuplicateID = errors.New("draftselect: duplicate request id")
	// ErrNegativeBudget 在预算为负时返回。
	ErrNegativeBudget = errors.New("draftselect: negative budget")
	// ErrLengthMismatch 在请求的候选词数量与概率数量不一致时返回。
	ErrLengthMismatch = errors.New("draftselect: words/probs length mismatch")
	// ErrInvalidProbability 在概率越界（不在 [0,1] 内）或非有限
	// （NaN、+Inf、-Inf）时返回。
	ErrInvalidProbability = errors.New("draftselect: invalid probability")
)

// requestData 保存按标识排序后的请求及其前缀连乘。
type requestData struct {
	id     string
	words  []string
	probs  []float64
	prefix []float64 // prefix[i] = probs[0]*...*probs[i]
}

// Engine 是已校验、按标识规范化的批次。它与具体预算无关，可以安全地
// 复用于同一输入的多个预算查询；其方法可被多个 goroutine 并发调用。
type Engine struct {
	requests []requestData // 按 id 升序，唯一
	// order 为全局贪心选择顺序：第 k 步选中的请求在 requests 中的下标。
	// 长度为所有候选位置总数。
	order []int
}

// Build 校验批次并构建引擎。校验内容包括：重复标识、词/概率长度不符、
// 概率越界或非有限。校验通过后，结果与请求在输入切片中的顺序无关。
func Build(requests []Request) (*Engine, error) {
	sorted := make([]Request, len(requests))
	copy(sorted, requests)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })

	data := make([]requestData, 0, len(sorted))
	for idx := range sorted {
		req := sorted[idx]
		if idx > 0 && req.ID == sorted[idx-1].ID {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateID, req.ID)
		}
		if len(req.Words) != len(req.Probs) {
			return nil, fmt.Errorf("%w: request %q has %d words but %d probabilities",
				ErrLengthMismatch, req.ID, len(req.Words), len(req.Probs))
		}
		rd := requestData{
			id:     req.ID,
			words:  req.Words,
			probs:  req.Probs,
			prefix: make([]float64, len(req.Probs)),
		}
		running := 1.0
		for pos, p := range req.Probs {
			if p < 0 || p > 1 || math.IsNaN(p) || math.IsInf(p, 0) {
				return nil, fmt.Errorf("%w: request %q position %d has %v",
					ErrInvalidProbability, req.ID, pos, p)
			}
			// 以 float64 直接做前缀连乘。浮点舍入保证乘以 <=1 的因子后
			// 结果不会严格变大，因此边际收益序列保持非递增；同时前缀闭合
			// 由选择结构强制保证，不会因极小概率下溢为 0 而被破坏。
			running *= p
			rd.prefix[pos] = running
		}
		data = append(data, rd)
	}

	return &Engine{requests: data, order: greedyOrder(data)}, nil
}

// candidate 是最小堆中的候选项：某条链当前的最前端位置。
type candidate struct {
	reqIdx int
	pos    int
	value  float64
}

type candidateHeap struct {
	items []candidate
}

func (h candidateHeap) Len() int { return len(h.items) }

// 我们需要“最大边际收益优先”，因此最小堆中把较大的 value 排在前面。
// value 相等时按请求标识、再按位置排序（请求已按标识升序存放，下标
// 顺序即标识顺序），得到确定且稳定的并列次序。
// 比较使用精确位相等（NaN 已在构建时拒绝），不使用 epsilon。
func (h candidateHeap) Less(i, j int) bool {
	a, b := h.items[i], h.items[j]
	if a.value != b.value {
		return a.value > b.value
	}
	if a.reqIdx != b.reqIdx {
		return a.reqIdx < b.reqIdx
	}
	return a.pos < b.pos
}

func (h candidateHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }

func (h *candidateHeap) Push(x any) { h.items = append(h.items, x.(candidate)) }

func (h *candidateHeap) Pop() any {
	old := h.items
	n := len(old)
	item := old[n-1]
	h.items = old[:n-1]
	return item
}

// greedyOrder 计算完整的贪心选择顺序。
//
// 每条链的边际收益 prefix[i] 非递增，故每个请求的总收益函数关于选中
// 长度为离散凹函数；在单位成本下，反复取当前最大的可行边际收益可得到
// 预算分配的最优解。只有选中位置 pos 后，位置 pos+1 才会进入候选，
// 因此前缀闭合是结构性强制的。
func greedyOrder(data []requestData) []int {
	total := 0
	h := &candidateHeap{items: make([]candidate, 0, len(data))}
	for i := range data {
		total += len(data[i].prefix)
		if len(data[i].prefix) > 0 {
			heap.Push(h, candidate{reqIdx: i, pos: 0, value: data[i].prefix[0]})
		}
	}
	order := make([]int, 0, total)
	for h.Len() > 0 {
		cur := heap.Pop(h).(candidate)
		order = append(order, cur.reqIdx)
		next := cur.pos + 1
		if next < len(data[cur.reqIdx].prefix) {
			heap.Push(h, candidate{
				reqIdx: cur.reqIdx,
				pos:    next,
				value:  data[cur.reqIdx].prefix[next],
			})
		}
	}
	return order
}

// NumRequests 返回批次中的请求数量。
func (e *Engine) NumRequests() int { return len(e.requests) }

// TotalCandidates 返回全部请求的候选位置总数（可用位置上限）。
func (e *Engine) TotalCandidates() int { return len(e.order) }

// Select 在给定预算下求解最优前缀分配。budget 为负时返回
// ErrNegativeBudget；候选充足时用满预算，候选不足时全部选择并在结果中
// 报告剩余额度。budget 为 0 时返回全部请求长度为 0 的结果。
func (e *Engine) Select(budget int) (*Result, error) {
	if budget < 0 {
		return nil, fmt.Errorf("%w: %d", ErrNegativeBudget, budget)
	}

	lengths := make([]int, len(e.requests))
	used := budget
	if used > len(e.order) {
		used = len(e.order)
	}
	for k := 0; k < used; k++ {
		lengths[e.order[k]]++
	}

	selections := make([]Selection, len(e.requests))
	// 总收益按全局规范顺序（请求标识升序，再按位置升序）直接累加各选中
	// 位置的前缀连乘，而不是先分组求各请求收益再加总。这样在并列方案
	// 总收益数学相等时，总和不因分组舍入而出现可观测的微小差异，结果
	// 对“选哪个同值位置”保持确定。
	var payoff float64
	for i, rd := range e.requests {
		length := lengths[i]
		sel := Selection{
			RequestID: rd.id,
			Length:    length,
			Positions: []int{},
			Words:     []string{},
		}
		var sum float64
		for pos := 0; pos < length; pos++ {
			sel.Positions = append(sel.Positions, pos)
			sel.Words = append(sel.Words, rd.words[pos])
			margin := rd.prefix[pos]
			sum += margin
			payoff += margin
		}
		sel.Payoff = sum
		selections[i] = sel
	}

	return &Result{
		Budget:     budget,
		Used:       used,
		Remaining:  budget - used,
		Payoff:     payoff,
		Selections: selections,
	}, nil
}

// SelectMany 对同一批次一次计算多个预算结果，按入参顺序返回。任一预算
// 为负时返回错误。所有结果共享同一份预计算贪心顺序，重复查询成本极低。
func (e *Engine) SelectMany(budgets []int) ([]*Result, error) {
	results := make([]*Result, len(budgets))
	for i, b := range budgets {
		r, err := e.Select(b)
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}
