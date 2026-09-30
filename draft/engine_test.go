package draft

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// approxEqual 比较两个浮点数在相对与绝对容差内是否相等，
// 避免不同求和顺序带来的末位差异。
func approxEqual(a, b, tol float64) bool {
	d := math.Abs(a - b)
	if d <= tol {
		return true
	}
	scale := math.Max(math.Abs(a), math.Abs(b))
	return d <= tol*scale
}

// chainsOf 把结果转成“每请求选择长度”的向量，按请求标识排序。
func lengthsOf(r *Result) map[string]int {
	m := make(map[string]int, len(r.Requests))
	for _, rr := range r.Requests {
		m[rr.ID] = rr.Length
	}
	return m
}

// prefixesClosed 验证每个请求都只选择从 1 开始的连续前缀。
func prefixesClosed(t *testing.T, r *Result) {
	t.Helper()
	for _, rr := range r.Requests {
		if len(rr.Positions) != rr.Length {
			t.Fatalf("request %s: positions len %d != length %d",
				rr.ID, len(rr.Positions), rr.Length)
		}
		for k, pos := range rr.Positions {
			if pos != k+1 {
				t.Fatalf("request %s: prefix broken at index %d, got position %d",
					rr.ID, k, pos)
			}
		}
	}
}

// budgetConserved 验证预算守恒：用量不超预算且不超候选总数，账实一致。
func budgetConserved(t *testing.T, r *Result, totalCandidates int) {
	t.Helper()
	sum := 0
	for _, rr := range r.Requests {
		sum += rr.Length
	}
	if sum != r.Used {
		t.Fatalf("sum of lengths %d != Used %d", sum, r.Used)
	}
	if r.Used+r.Remaining != r.Budget {
		t.Fatalf("used %d + remaining %d != budget %d",
			r.Used, r.Remaining, r.Budget)
	}
	if r.Used > r.Budget || r.Used > totalCandidates {
		t.Fatalf("used %d exceeds budget %d or candidates %d",
			r.Used, r.Budget, totalCandidates)
	}
	wantUsed := r.Budget
	if wantUsed > totalCandidates {
		wantUsed = totalCandidates
	}
	if r.Used != wantUsed {
		t.Fatalf("used = %d, want %d (budget %d, candidates %d)",
			r.Used, wantUsed, r.Budget, totalCandidates)
	}
}

// bruteForce 是小规模穷举参考：枚举每个请求的所有合法前缀长度，
// 求预算内的最大总收益，以及最优解中按请求标识排序的字典序最大长度向量
// （与引擎的“标识升序、位置升序”并列规则对应）。仅用于测试验证。
func bruteForce(reqs []Request, budget int) (float64, map[string]int) {
	sorted := append([]Request(nil), reqs...)
	slices.SortFunc(sorted, func(a, b Request) int {
		if a.ID < b.ID {
			return -1
		}
		if a.ID > b.ID {
			return 1
		}
		return 0
	})

	// pref[i][k] = 请求 i 选择前 k 个位置时的收益和。
	pref := make([][]float64, len(sorted))
	for i, req := range sorted {
		pref[i] = make([]float64, len(req.AcceptProb)+1)
		prod := 1.0
		for k, p := range req.AcceptProb {
			prod *= p
			pref[i][k+1] = pref[i][k] + prod
		}
	}

	var (
		best    = math.Inf(-1)
		bestLen map[string]int
		cur     = make([]int, len(sorted))
	)

	var dfs func(i, used int, payoff float64)
	dfs = func(i, used int, payoff float64) {
		if i == len(sorted) {
			if payoff > best+1e-15 {
				best = payoff
				bestLen = map[string]int{}
				for j, req := range sorted {
					bestLen[req.ID] = cur[j]
				}
				return
			}
			if approxEqual(payoff, best, 1e-15) {
				// 并列时取字典序（按标识排序）更大的长度向量。
				for j := range sorted {
					if cur[j] > bestLen[sorted[j].ID] {
						for k, req := range sorted {
							bestLen[req.ID] = cur[k]
						}
						break
					}
					if cur[j] < bestLen[sorted[j].ID] {
						break
					}
				}
			}
			return
		}
		maxLen := len(sorted[i].AcceptProb)
		if remain := budget - used; remain < maxLen {
			maxLen = remain
		}
		for ln := 0; ln <= maxLen; ln++ {
			cur[i] = ln
			dfs(i+1, used+ln, payoff+pref[i][ln])
		}
	}
	dfs(0, 0, 0)
	return best, bestLen
}

// TestOptimalByExhaustive 对大量小规模随机批次做穷举对照，
// 断言非均匀概率下引擎达到最优收益、长度向量并列选择一致、
// 前缀闭合且预算守恒。
func TestOptimalByExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(20261001))

	for iter := 0; iter < 400; iter++ {
		n := 1 + rng.Intn(4)
		reqs := make([]Request, n)
		total := 0
		for i := range reqs {
			ln := rng.Intn(5) // 每请求 0..4 个候选
			req := Request{
				ID:         fmt.Sprintf("r%02d", i),
				Words:      make([]string, ln),
				AcceptProb: make([]float64, ln),
			}
			for k := 0; k < ln; k++ {
				req.Words[k] = "w"
				// 含 0、1 与非均匀分布，刻意让边际收益形状多样。
				switch rng.Intn(6) {
				case 0:
					req.AcceptProb[k] = 0
				case 1:
					req.AcceptProb[k] = 1
				default:
					req.AcceptProb[k] = rng.Float64()
				}
			}
			reqs[i] = req
			total += ln
		}

		eng, err := NewEngine(reqs)
		if err != nil {
			t.Fatalf("iter %d: unexpected error: %v", iter, err)
		}
		if eng.TotalCandidates() != total {
			t.Fatalf("iter %d: total candidates %d, want %d",
				iter, eng.TotalCandidates(), total)
		}

		budget := rng.Intn(total + 4) // 含 0 与候选不足情形
		res, err := eng.Select(budget)
		if err != nil {
			t.Fatalf("iter %d: select error: %v", iter, err)
		}

		bestPayoff, bestLen := bruteForce(reqs, budget)
		if !approxEqual(res.TotalPayoff, bestPayoff, 1e-12) {
			t.Fatalf("iter %d budget %d: payoff %.18g, optimal %.18g",
				iter, budget, res.TotalPayoff, bestPayoff)
		}
		gotLen := lengthsOf(res)
		for id, want := range bestLen {
			if gotLen[id] != want {
				t.Fatalf("iter %d budget %d: request %s length %d, want %d (tie rule)",
					iter, budget, id, gotLen[id], want)
			}
		}
		prefixesClosed(t, res)
		budgetConserved(t, res, total)

		// 逐请求收益之和必须等于总收益。
		sum := 0.0
		for _, rr := range res.Requests {
			sum += rr.Payoff
		}
		if !approxEqual(sum, res.TotalPayoff, 1e-15) {
			t.Fatalf("iter %d: per-request payoff sum %.18g != total %.18g",
				iter, sum, res.TotalPayoff)
		}
	}
}

// TestNonUniformOptimal 用非均匀概率验证：收益是条件概率连乘，
// 贪心按边际连乘收益分配，并断言真实的选择长度、位置与收益。
func TestNonUniformOptimal(t *testing.T) {
	reqs := []Request{
		{ID: "A", Words: []string{"a1", "a2", "a3"},
			AcceptProb: []float64{0.9, 0.8, 0.2}},
		{ID: "B", Words: []string{"b1", "b2"},
			AcceptProb: []float64{0.5, 0.99}},
	}
	eng, err := NewEngine(reqs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 边际连乘收益：
	// A: 0.9, 0.72, 0.144
	// B: 0.5, 0.495
	// 预算 3 的前三名为 0.9(A1)、0.72(A2)、0.5(B1)。
	res, err := eng.Select(3)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	got := lengthsOf(res)
	if got["A"] != 2 || got["B"] != 1 {
		t.Fatalf("lengths = %v, want A=2 B=1", got)
	}
	if !approxEqual(res.TotalPayoff, 0.9+0.72+0.5, 1e-12) {
		t.Fatalf("total payoff = %.18g, want %.18g",
			res.TotalPayoff, 0.9+0.72+0.5)
	}
	if res.Used != 3 || res.Remaining != 0 {
		t.Fatalf("used=%d remaining=%d, want 3/0", res.Used, res.Remaining)
	}
	for _, rr := range res.Requests {
		wantPayoff := map[string]float64{"A": 0.9 + 0.72, "B": 0.5}[rr.ID]
		if !approxEqual(rr.Payoff, wantPayoff, 1e-12) {
			t.Fatalf("request %s payoff %.18g, want %.18g",
				rr.ID, rr.Payoff, wantPayoff)
		}
	}

	// 预算 4：第四名 0.495(B2) 先于 0.144(A3)。
	res4, err := eng.Select(4)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	got4 := lengthsOf(res4)
	if got4["A"] != 2 || got4["B"] != 2 {
		t.Fatalf("lengths = %v, want A=2 B=2", got4)
	}

	// 预算 5 全取。
	res5, err := eng.Select(5)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	got5 := lengthsOf(res5)
	if got5["A"] != 3 || got5["B"] != 2 {
		t.Fatalf("lengths = %v, want A=3 B=2", got5)
	}
	prefixesClosed(t, res5)
}

// TestStableTie 相同收益时按请求标识、再按位置定序；打乱输入顺序结果不变。
func TestStableTie(t *testing.T) {
	build := func() []Request {
		return []Request{
			{ID: "zeta", Words: []string{"z1", "z2"}, AcceptProb: []float64{0.5, 1}},
			{ID: "alpha", Words: []string{"a1", "a2"}, AcceptProb: []float64{0.5, 1}},
			{ID: "mid", Words: []string{"m1", "m2"}, AcceptProb: []float64{0.5, 1}},
		}
	}

	eng1, err := NewEngine(build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	shuffled := build()
	shuffled[0], shuffled[1], shuffled[2] = shuffled[2], shuffled[0], shuffled[1]
	eng2, err := NewEngine(shuffled)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, budget := range []int{0, 1, 2, 3, 4, 5, 6, 10} {
		r1, err := eng1.Select(budget)
		if err != nil {
			t.Fatalf("select error: %v", err)
		}
		r2, err := eng2.Select(budget)
		if err != nil {
			t.Fatalf("select error: %v", err)
		}

		ids := make([]string, len(r1.Requests))
		for i, rr := range r1.Requests {
			ids[i] = rr.ID
			if r2.Requests[i].ID != rr.ID ||
				r2.Requests[i].Length != rr.Length ||
				!approxEqual(r2.Requests[i].Payoff, rr.Payoff, 1e-15) {
				t.Fatalf("budget %d: results differ under input reordering:\n%+v\n%+v",
					budget, r1.Requests, r2.Requests)
			}
		}
		wantOrder := []string{"alpha", "mid", "zeta"}
		if !slices.Equal(ids, wantOrder) {
			t.Fatalf("budget %d: result IDs = %v, want sorted %v", budget, ids, wantOrder)
		}

		// 三链边际收益同为 0.5、0.5（p2=1）。预算 4 时全部最优解并列，
		// 稳定规则取按标识排序的字典序最大长度向量：alpha=2 mid=2 zeta=0。
		if budget == 4 {
			got := lengthsOf(r1)
			if got["alpha"] != 2 || got["mid"] != 2 || got["zeta"] != 0 {
				t.Fatalf("budget 4 lengths = %v, want alpha=2 mid=2 zeta=0", got)
			}
		}
		prefixesClosed(t, r1)
		budgetConserved(t, r1, 6)
	}
}

// TestInsufficientCandidates 候选不足时全部选择，报告剩余额度。
func TestInsufficientCandidates(t *testing.T) {
	reqs := []Request{
		{ID: "x", Words: []string{"w"}, AcceptProb: []float64{0.7}},
		{ID: "y", Words: []string{}, AcceptProb: []float64{}},
		{ID: "z", Words: []string{"w1", "w2"}, AcceptProb: []float64{0.6, 0.5}},
	}
	eng, err := NewEngine(reqs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if eng.TotalCandidates() != 3 {
		t.Fatalf("total candidates = %d, want 3", eng.TotalCandidates())
	}

	res, err := eng.Select(10)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	if res.Used != 3 || res.Remaining != 7 {
		t.Fatalf("used=%d remaining=%d, want 3/7", res.Used, res.Remaining)
	}
	got := lengthsOf(res)
	if got["x"] != 1 || got["y"] != 0 || got["z"] != 2 {
		t.Fatalf("lengths = %v, want x=1 y=0 z=2", got)
	}
	// 0.7, 0.6, 0.3(=0.6*0.5)
	if !approxEqual(res.TotalPayoff, 0.7+0.6+0.3, 1e-12) {
		t.Fatalf("total payoff = %.18g, want 1.6", res.TotalPayoff)
	}
	for _, rr := range res.Requests {
		if rr.ID == "y" {
			if rr.Positions != nil {
				t.Fatalf("empty candidate request should have nil/empty positions")
			}
			if rr.Payoff != 0 {
				t.Fatalf("empty candidate request payoff = %v, want 0", rr.Payoff)
			}
		}
	}
	prefixesClosed(t, res)
	budgetConserved(t, res, 3)

	// 再用更小预算查询同一引擎，确认引擎可复用且互不影响。
	res2, err := eng.Select(2)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	if res2.Used != 2 || res2.Remaining != 0 {
		t.Fatalf("used=%d remaining=%d, want 2/0", res2.Used, res2.Remaining)
	}
}

// TestEmptyBatchAndZeroBudget 空批次与零预算都要有明确结果。
func TestEmptyBatchAndZeroBudget(t *testing.T) {
	eng, err := NewEngine(nil)
	if err != nil {
		t.Fatalf("empty batch should be valid, got %v", err)
	}
	res, err := eng.Select(0)
	if err != nil {
		t.Fatalf("zero budget should be valid, got %v", err)
	}
	if res.Used != 0 || res.Remaining != 0 || res.TotalPayoff != 0 {
		t.Fatalf("empty/zero result = %+v, want all zero", res)
	}
	if len(res.Requests) != 0 {
		t.Fatalf("empty batch requests = %d, want 0", len(res.Requests))
	}

	// 空批次但给正预算：候选不足，额度全部剩余。
	res5, err := eng.Select(5)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	if res5.Used != 0 || res5.Remaining != 5 {
		t.Fatalf("used=%d remaining=%d, want 0/5", res5.Used, res5.Remaining)
	}

	// 非空批次零预算：每请求长度 0、位置为空。
	eng2, err := NewEngine([]Request{
		{ID: "a", Words: []string{"w"}, AcceptProb: []float64{0.9}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	res0, err := eng2.Select(0)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	if res0.Used != 0 || res0.Requests[0].Length != 0 ||
		len(res0.Requests[0].Positions) != 0 || res0.Requests[0].Payoff != 0 {
		t.Fatalf("zero budget result = %+v, want empty selection", res0)
	}
}

// TestMidZeroProbability 中途出现零概率：其后位置收益全部为 0，
// 前缀仍可在预算充裕时被选中，且零尾不会抢在正收益位置之前。
func TestMidZeroProbability(t *testing.T) {
	reqs := []Request{
		// 第二个位置条件概率为 0，故 q2=q3=0。
		{ID: "zero-tail", Words: []string{"w1", "w2", "w3"},
			AcceptProb: []float64{0.8, 0.0, 0.9}},
		{ID: "positive", Words: []string{"w1", "w2"},
			AcceptProb: []float64{0.2, 0.5}}, // q = 0.2, 0.1
	}
	eng, err := NewEngine(reqs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	res, err := eng.Select(3)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	got := lengthsOf(res)
	// 顺序：0.8(z.1)、0.2(p.1)、0.1(p.2)，零收益位置排在最后。
	if got["zero-tail"] != 1 || got["positive"] != 2 {
		t.Fatalf("lengths = %v, want zero-tail=1 positive=2", got)
	}

	resFull, err := eng.Select(5)
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	if resFull.Used != 5 || resFull.Remaining != 0 {
		t.Fatalf("used=%d remaining=%d, want 5/0", resFull.Used, resFull.Remaining)
	}
	gotFull := lengthsOf(resFull)
	if gotFull["zero-tail"] != 3 || gotFull["positive"] != 2 {
		t.Fatalf("lengths = %v, want zero-tail=3 positive=2", gotFull)
	}
	if !approxEqual(resFull.TotalPayoff, 0.8+0.2+0.1, 1e-12) {
		t.Fatalf("full total payoff = %.18g, want 1.1", resFull.TotalPayoff)
	}
	prefixesClosed(t, resFull)
	budgetConserved(t, resFull, 5)
}

// TestInvalidInputs 覆盖重复标识、长度不符、越界与非有限概率、负预算。
func TestInvalidInputs(t *testing.T) {
	cases := []struct {
		name string
		reqs []Request
	}{
		{
			name: "duplicate id",
			reqs: []Request{
				{ID: "a", Words: []string{"w"}, AcceptProb: []float64{0.5}},
				{ID: "a", Words: []string{"w"}, AcceptProb: []float64{0.5}},
			},
		},
		{
			name: "empty id",
			reqs: []Request{
				{ID: "", Words: []string{"w"}, AcceptProb: []float64{0.5}},
			},
		},
		{
			name: "length mismatch",
			reqs: []Request{
				{ID: "a", Words: []string{"w1", "w2"}, AcceptProb: []float64{0.5}},
			},
		},
		{
			name: "probability negative",
			reqs: []Request{
				{ID: "a", Words: []string{"w"}, AcceptProb: []float64{-0.01}},
			},
		},
		{
			name: "probability above one",
			reqs: []Request{
				{ID: "a", Words: []string{"w"}, AcceptProb: []float64{1.0001}},
			},
		},
		{
			name: "probability NaN",
			reqs: []Request{
				{ID: "a", Words: []string{"w"}, AcceptProb: []float64{math.NaN()}},
			},
		},
		{
			name: "probability +Inf",
			reqs: []Request{
				{ID: "a", Words: []string{"w"}, AcceptProb: []float64{math.Inf(1)}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewEngine(tc.reqs); err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
		})
	}

	eng, err := NewEngine([]Request{
		{ID: "a", Words: []string{"w"}, AcceptProb: []float64{0.5}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := eng.Select(-1); err == nil {
		t.Fatalf("expected error for negative budget")
	}
}

// TestLongTinyPrefix 验证极长前缀的极小连乘概率不会因 float64 下溢
// 而破坏排序与前缀约束：两条长链 q 在深处同时下溢为 0，
// 引擎改用对数和仍保持真实收益次序。
func TestLongTinyPrefix(t *testing.T) {
	const n = 2200
	mk := func(id string, p float64) Request {
		req := Request{ID: id, Words: make([]string, n), AcceptProb: make([]float64, n)}
		for k := range req.AcceptProb {
			req.AcceptProb[k] = p
		}
		return req
	}
	// zeta 每位置概率更高，深处其连乘收益始终严格大于 alpha。
	reqs := []Request{mk("alpha", 0.49), mk("zeta", 0.5)}
	eng, err := NewEngine(reqs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 找到一个 q 已经下溢为 0、但 logq 仍为有限值的位置，确认回退路径真实生效。
	foundUnderflow := false
	for _, s := range eng.order {
		if s.q == 0 && !math.IsInf(s.logq, 0) {
			foundUnderflow = true
			break
		}
	}
	if !foundUnderflow {
		t.Fatalf("expected some deep positions to underflow q to 0 with finite logq")
	}

	res, err := eng.Select(n) // 2200 预算，两链各 2200 候选
	if err != nil {
		t.Fatalf("select error: %v", err)
	}
	if res.Used != n || res.Remaining != 0 {
		t.Fatalf("used=%d remaining=%d, want %d/0", res.Used, res.Remaining, n)
	}
	got := lengthsOf(res)
	la, lz := got["alpha"], got["zeta"]
	if !(lz > la) {
		t.Fatalf("higher-probability chain should be deeper: alpha=%d zeta=%d", la, lz)
	}
	if la+lz != n {
		t.Fatalf("lengths sum %d != budget %d", la+lz, n)
	}
	prefixesClosed(t, res)
	budgetConserved(t, res, 2*n)

	t.Logf("long-prefix allocation under budget %d: alpha=%d zeta=%d", n, la, lz)
}

// TestMultipleBudgets 同一输入计算多个预算，结果单调扩展、前缀闭合、预算守恒。
func TestMultipleBudgets(t *testing.T) {
	reqs := []Request{
		{ID: "a", Words: []string{"1", "2", "3"}, AcceptProb: []float64{0.9, 0.7, 0.6}},
		{ID: "b", Words: []string{"1", "2"}, AcceptProb: []float64{0.8, 0.5}},
	}
	eng, err := NewEngine(reqs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	prev := map[string]int{}
	prevPayoff := 0.0
	for _, budget := range []int{0, 1, 2, 3, 4, 5, 6, 7, 100} {
		res, err := eng.Select(budget)
		if err != nil {
			t.Fatalf("budget %d: %v", budget, err)
		}
		got := lengthsOf(res)
		for id, ln := range got {
			if ln < prev[id] {
				t.Fatalf("budget %d shrank request %s: %d -> %d",
					budget, id, prev[id], ln)
			}
		}
		if res.TotalPayoff+1e-15 < prevPayoff {
			t.Fatalf("budget %d payoff decreased: %.18g < %.18g",
				budget, res.TotalPayoff, prevPayoff)
		}
		prefixesClosed(t, res)
		budgetConserved(t, res, 5)
		prev = got
		prevPayoff = res.TotalPayoff
	}
}
