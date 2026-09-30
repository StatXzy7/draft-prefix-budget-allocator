// Package draft 实现一个批次草稿选择引擎：在固定核验位置预算内，
// 为一批请求分配从各自起点开始的连续草稿前缀，使本轮预期接受词数最大。
//
// 每个请求只能选择零个词，或选择从第 1 个位置起的完整前缀，不允许跳过
// 未选择的位置。位置 k 的收益是“此前词全部被接受”条件下接受概率的连乘积
// p1*p2*...*pk（不是单位置概率本身）；总收益为全部选中位置收益之和。
package draft

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
)

// Request 描述批次中的一个请求。
//
// AcceptProb[k] 是在第 1..k 个词全部被接受的条件下，第 k+1 个词
// （即 Words[k]，零基索引）被接受的条件概率。当某个条件概率为 0 时，
// 该位置及其后位置的连乘收益均为 0。
type Request struct {
	ID         string
	Words      []string
	AcceptProb []float64
}

// RequestResult 是单个请求在某个预算下的选择结果。
type RequestResult struct {
	ID        string
	Length    int     // 选中的前缀长度（位置个数）
	Positions []int   // 选中的位置编号（从 1 开始）；长度为 0 时为空
	Payoff    float64 // 该请求已选中位置的连乘收益之和
}

// Result 是整个批次在某个预算下的选择结果。
type Result struct {
	Budget      int
	Used        int // 实际消耗的预算；候选不足时小于 Budget
	Remaining   int // 剩余额度
	TotalPayoff float64
	Requests    []RequestResult // 按请求标识排序，与输入顺序无关
}

// slot 是贪心合并序列中的一个候选项：某请求前缀的一个位置。
type slot struct {
	chain int     // 在按标识排序后的链下标
	id    string  // 请求标识，用于稳定并列
	pos   int     // 位置编号（从 1 开始）
	q     float64 // 连乘收益 p1*...*p_pos
	logq  float64 // 收益对数之和，q 下溢为 0 时仍可正确排序
}

// chain 保存单个请求预先计算好的全部位置连乘收益。
type chain struct {
	id   string
	q    []float64
	logq []float64
}

// Engine 持有已校验并预先计算的批次，可对同一批次反复查询不同预算。
// 每个请求的全部位置构成一个非递增的边际收益序列；使用最小堆合并这些
// 序列，即得到整批位置边际收益的非递增顺序，该贪心分配对这类可分凹
// （前缀闭合）资源分配问题是最优的。
type Engine struct {
	chains []chain
	order  []slot // 边际收益从大到小的贪心选择顺序
	total  int    // 全部候选位置总数
}

// NewEngine 校验输入、预先计算连乘收益并构建贪心选择顺序。
//
// 拒绝情形：
//   - 空请求标识或重复标识；
//   - 候选词数与条件概率数不一致；
//   - 概率越界（小于 0 或大于 1）或非有限（NaN/Inf）。
//
// 空批次（没有请求）与空候选（请求存在但没有词）都是合法输入。
func NewEngine(requests []Request) (*Engine, error) {
	chains := make([]chain, len(requests))
	seen := make(map[string]struct{}, len(requests))
	total := 0

	for i, req := range requests {
		if req.ID == "" {
			return nil, fmt.Errorf("draft: request at index %d has empty ID", i)
		}
		if _, dup := seen[req.ID]; dup {
			return nil, fmt.Errorf("draft: duplicate request ID %q", req.ID)
		}
		seen[req.ID] = struct{}{}

		if len(req.Words) != len(req.AcceptProb) {
			return nil, fmt.Errorf(
				"draft: request %q: %d words but %d probabilities",
				req.ID, len(req.Words), len(req.AcceptProb),
			)
		}

		n := len(req.AcceptProb)
		c := chain{id: req.ID, q: make([]float64, n), logq: make([]float64, n)}
		prod := 1.0
		logSum := 0.0
		zeroed := false
		for k, p := range req.AcceptProb {
			if math.IsNaN(p) || math.IsInf(p, 0) || p < 0.0 || p > 1.0 {
				return nil, fmt.Errorf(
					"draft: request %q position %d: invalid probability %v",
					req.ID, k+1, p,
				)
			}
			if zeroed {
				c.q[k] = 0
				c.logq[k] = math.Inf(-1)
				continue
			}
			if p == 0.0 {
				zeroed = true
				prod = 0
				logSum = math.Inf(-1)
				c.q[k] = 0
				c.logq[k] = math.Inf(-1)
				continue
			}
			prod *= p
			logSum += math.Log(p)
			c.q[k] = prod
			c.logq[k] = logSum
		}
		chains[i] = c
		total += n
	}

	sort.Slice(chains, func(i, j int) bool { return chains[i].id < chains[j].id })

	e := &Engine{chains: chains, total: total, order: make([]slot, 0, total)}

	h := make(priorityHeap, 0, len(chains))
	for i := range chains {
		if len(chains[i].q) > 0 {
			h = append(h, slot{
				chain: i,
				id:    chains[i].id,
				pos:   1,
				q:     chains[i].q[0],
				logq:  chains[i].logq[0],
			})
		}
	}
	heap.Init(&h)

	for h.Len() > 0 {
		top := heap.Pop(&h).(slot)
		e.order = append(e.order, top)
		c := e.chains[top.chain]
		if top.pos < len(c.q) {
			heap.Push(&h, slot{
				chain: top.chain,
				id:    top.id,
				pos:   top.pos + 1,
				q:     c.q[top.pos], // 零基索引：位置 pos+1 对应下标 pos
				logq:  c.logq[top.pos],
			})
		}
	}

	return e, nil
}

// TotalCandidates 返回批次中全部候选位置的总数（所有预算的实际上限）。
func (e *Engine) TotalCandidates() int { return e.total }

// Select 在给定预算下执行选择。预算为负时返回错误；预算为 0 时结果为空。
// 候选充足时用满预算；候选不足时全部选择并报告剩余额度。
// Select 只读，可对同一引擎并发调用。
func (e *Engine) Select(budget int) (*Result, error) {
	if budget < 0 {
		return nil, fmt.Errorf("draft: negative budget %d", budget)
	}

	used := budget
	if used > e.total {
		used = e.total
	}

	length := make([]int, len(e.chains))
	payoff := make([]float64, len(e.chains))
	totalPayoff := 0.0
	for k := 0; k < used; k++ {
		s := e.order[k]
		length[s.chain]++
		payoff[s.chain] += s.q
		totalPayoff += s.q
	}

	results := make([]RequestResult, len(e.chains))
	for i, c := range e.chains {
		r := RequestResult{
			ID:     c.id,
			Length: length[i],
			Payoff: payoff[i],
		}
		if length[i] > 0 {
			r.Positions = make([]int, length[i])
			for k := range r.Positions {
				r.Positions[k] = k + 1
			}
		}
		results[i] = r
	}

	return &Result{
		Budget:      budget,
		Used:        used,
		Remaining:   budget - used,
		TotalPayoff: totalPayoff,
		Requests:    results,
	}, nil
}

// priorityHeap 按边际收益排序：收益大者优先；收益相等时，按请求标识升序、
// 再按位置升序，保证并列结果与输入顺序无关且完全确定。q 因长连乘下溢为
// 0 时改用 logq 比较，保留前缀间真实的收益次序与前缀闭合约束。
type priorityHeap []slot

func (h priorityHeap) Len() int { return len(h) }

func (h priorityHeap) Less(i, j int) bool {
	a, b := h[i], h[j]
	if a.q != b.q {
		return a.q > b.q
	}
	if a.logq != b.logq {
		return a.logq > b.logq
	}
	if a.id != b.id {
		return a.id < b.id
	}
	return a.pos < b.pos
}

func (h priorityHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *priorityHeap) Push(x any) { *h = append(*h, x.(slot)) }

func (h *priorityHeap) Pop() any {
	old := *h
	n := len(old)
	s := old[n-1]
	*h = old[:n-1]
	return s
}
