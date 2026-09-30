package draftselect_test

import (
	"errors"
	"math"
	"math/big"
	"math/rand"
	"reflect"
	"testing"

	ds "draftselect"
)

// ---- 小规模穷举参考（仅用于测试，不参与正常引擎实现） ----
//
// exhaustive 独立地重算前缀连乘，并枚举每个请求所有可行的选中长度
// （0..候选数，且总长度不超过预算），找出总收益最大的分配。
//
// 关键：参考对引擎实际使用的 float64 边际值（输入概率经 float64
// 前缀连乘得到）用 math/big.Rat 做精确有理求和，因此“不同选中子集
// 在 float64 下因加法不可结合产生 1-ulp 差”不会伪造或掩盖并列。
// 精确总收益并列时，选择按请求标识升序、长度向量字典序最大的分配，
// 这与引擎“同收益按请求标识、再按位置”的稳定并列规则一致。
//
// 仅适用于小规模输入（测试中每链 <=5、请求 <=5）。
func exhaustive(t *testing.T, in []ds.Request, budget int) ([]int, *big.Rat) {
	t.Helper()
	reqs := append([]ds.Request(nil), in...)
	// 按标识排序（与引擎规范化方式一致），用稳定排序保持确定性。
	for i := 1; i < len(reqs); i++ {
		for j := i; j > 0 && reqs[j].ID < reqs[j-1].ID; j-- {
			reqs[j], reqs[j-1] = reqs[j-1], reqs[j]
		}
	}

	type chain struct {
		// 与引擎完全相同的 float64 前缀连乘，再逐项精确化为有理数。
		margin []*big.Rat
	}
	chains := make([]chain, len(reqs))
	for i, r := range reqs {
		if len(r.Words) != len(r.Probs) {
			t.Fatalf("exhaustive: length mismatch for %q", r.ID)
		}
		product := 1.0
		margin := make([]*big.Rat, len(r.Probs))
		for pos, p := range r.Probs {
			product *= p
			margin[pos] = new(big.Rat).SetFloat64(product)
			if margin[pos] == nil {
				t.Fatalf("exhaustive: non-finite prefix at %q[%d]", r.ID, pos)
			}
		}
		chains[i] = chain{margin: margin}
	}

	bestRat := new(big.Rat)
	haveBest := false
	bestLens := make([]int, len(reqs))
	cur := make([]int, len(reqs))

	// 到达叶节点后对 float64 边际值做精确有理求和（与加法顺序无关）。
	totalRat := func(lens []int) *big.Rat {
		total := new(big.Rat)
		for i, ch := range chains {
			for pos := 0; pos < lens[i]; pos++ {
				total.Add(total, ch.margin[pos])
			}
		}
		return total
	}

	var dfs func(i, remaining int)
	dfs = func(i, remaining int) {
		if i == len(reqs) {
			total := totalRat(cur)
			if !haveBest || total.Cmp(bestRat) > 0 {
				bestRat.Set(total)
				haveBest = true
				copy(bestLens, cur)
				return
			}
			if total.Cmp(bestRat) == 0 && lexGreater(cur, bestLens) {
				copy(bestLens, cur)
			}
			return
		}
		maxLen := len(chains[i].margin)
		if remaining < maxLen {
			maxLen = remaining
		}
		for k := 0; k <= maxLen; k++ {
			cur[i] = k
			dfs(i+1, remaining-k)
		}
		cur[i] = 0
	}
	dfs(0, budget)
	return bestLens, bestRat
}

func lexGreater(a, b []int) bool {
	for i := range a {
		if a[i] != b[i] {
			return a[i] > b[i]
		}
	}
	return false
}

func mustBuild(t *testing.T, reqs []ds.Request) *ds.Engine {
	t.Helper()
	e, err := ds.Build(reqs)
	if err != nil {
		t.Fatalf("Build: unexpected error: %v", err)
	}
	return e
}

func floatClose(a, b float64) bool { return math.Abs(a-b) <= 1e-12 }

func ratFloat(r *big.Rat) float64 {
	f, _ := r.Float64()
	return f
}

func selByID(r *ds.Result) map[string]ds.Selection {
	m := make(map[string]ds.Selection, len(r.Selections))
	for _, s := range r.Selections {
		m[s.RequestID] = s
	}
	return m
}

func assertResultAgainstReference(t *testing.T, in []ds.Request, budget int) *ds.Result {
	t.Helper()
	e := mustBuild(t, in)
	r, err := e.Select(budget)
	if err != nil {
		t.Fatalf("Select(%d): %v", budget, err)
	}
	expLens, expPayoff := exhaustive(t, in, budget)

	if len(r.Selections) != len(expLens) {
		t.Fatalf("budget %d: got %d selections, want %d", budget, len(r.Selections), len(expLens))
	}
	for i, s := range r.Selections {
		if s.Length != expLens[i] {
			t.Fatalf("budget %d, request %q: length=%d, want %d (optimal reference)",
				budget, s.RequestID, s.Length, expLens[i])
		}
	}
	if !floatClose(r.Payoff, ratFloat(expPayoff)) {
		t.Fatalf("budget %d: payoff=%.17g, want optimal %.17g", budget, r.Payoff, ratFloat(expPayoff))
	}
	assertPrefixClosureAndConservation(t, in, r, e.TotalCandidates())
	return r
}

// assertPrefixClosureAndConservation 校验：
//   - 前缀闭合：每请求选中位置恰为 0..Length-1，词与输入候选一一对应；
//   - 收益真实：独立从输入概率重算前缀连乘，逐请求、逐总计核对；
//   - 预算守恒：Used==各长度之和==min(Budget,候选总数)，Used+Remaining==Budget；
//   - 数值健康：所有收益有限且非负。
func assertPrefixClosureAndConservation(t *testing.T, in []ds.Request, r *ds.Result, totalCandidates int) {
	t.Helper()
	if r.Used < 0 || r.Remaining < 0 {
		t.Fatalf("negative used/remaining: %d/%d", r.Used, r.Remaining)
	}
	inputByID := make(map[string]ds.Request, len(in))
	for _, req := range in {
		inputByID[req.ID] = req
	}
	sumLens := 0
	var sumPayoff float64
	for _, s := range r.Selections {
		req, ok := inputByID[s.RequestID]
		if !ok {
			t.Fatalf("selection for unknown request %q", s.RequestID)
		}
		sumLens += s.Length
		if s.Length > len(req.Probs) {
			t.Fatalf("request %q: length %d exceeds %d candidates", s.RequestID, s.Length, len(req.Probs))
		}
		if len(s.Positions) != s.Length || len(s.Words) != s.Length {
			t.Fatalf("request %q: positions/words length %d/%d != Length %d",
				s.RequestID, len(s.Positions), len(s.Words), s.Length)
		}
		var perPayoff float64
		product := 1.0
		for pos := 0; pos < s.Length; pos++ {
			if s.Positions[pos] != pos {
				t.Fatalf("request %q: positions=%v, want contiguous prefix 0..%d",
					s.RequestID, s.Positions, s.Length-1)
			}
			if s.Words[pos] != req.Words[pos] {
				t.Fatalf("request %q position %d: word=%q, want %q",
					s.RequestID, pos, s.Words[pos], req.Words[pos])
			}
			product *= req.Probs[pos]
			perPayoff += product
		}
		if math.IsNaN(s.Payoff) || math.IsInf(s.Payoff, 0) || s.Payoff < 0 {
			t.Fatalf("request %q: bad payoff %v", s.RequestID, s.Payoff)
		}
		if !floatClose(s.Payoff, perPayoff) {
			t.Fatalf("request %q: reported payoff %.17g != recomputed %.17g",
				s.RequestID, s.Payoff, perPayoff)
		}
		sumPayoff += s.Payoff
	}
	if r.Used != sumLens {
		t.Fatalf("Used=%d but sum of lengths=%d", r.Used, sumLens)
	}
	wantUsed := r.Budget
	if wantUsed > totalCandidates {
		wantUsed = totalCandidates
	}
	if r.Used != wantUsed {
		t.Fatalf("Used=%d, want min(%d,%d)=%d", r.Used, r.Budget, totalCandidates, wantUsed)
	}
	if r.Used+r.Remaining != r.Budget {
		t.Fatalf("conservation broken: %d+%d != %d", r.Used, r.Remaining, r.Budget)
	}
	if !floatClose(r.Payoff, sumPayoff) {
		t.Fatalf("total payoff %.17g != sum of per-request payoffs %.17g", r.Payoff, sumPayoff)
	}
	if math.IsNaN(r.Payoff) || math.IsInf(r.Payoff, 0) || r.Payoff < 0 {
		t.Fatalf("bad total payoff %v", r.Payoff)
	}
}

// TestNonUniformOptimal 用全部为二进制可精确表示分数的非均匀概率，
// 逐预算断言引擎达到最优收益（手算值），并与穷举参考一致。
func TestNonUniformOptimal(t *testing.T) {
	// a: 前缀连乘 [1, 1/2, 1/4]
	// b: 前缀连乘 [1/2, 1/4]
	// c: 前缀连乘 [3/4, 3/8]
	reqs := []ds.Request{
		{ID: "a", Words: []string{"a0", "a1", "a2"}, Probs: []float64{1, 0.5, 0.5}},
		{ID: "b", Words: []string{"b0", "b1"}, Probs: []float64{0.5, 0.5}},
		{ID: "c", Words: []string{"c0", "c1"}, Probs: []float64{0.75, 0.5}},
	}
	e := mustBuild(t, reqs)
	if e.TotalCandidates() != 7 {
		t.Fatalf("total candidates=%d, want 7", e.TotalCandidates())
	}

	want := []struct {
		budget    int
		used      int
		remaining int
		lens      map[string]int
		payoff    float64
	}{
		{0, 0, 0, map[string]int{"a": 0, "b": 0, "c": 0}, 0},
		{1, 1, 0, map[string]int{"a": 1, "b": 0, "c": 0}, 1},
		{2, 2, 0, map[string]int{"a": 1, "b": 0, "c": 1}, 1.75},
		{3, 3, 0, map[string]int{"a": 2, "b": 0, "c": 1}, 2.25},
		{4, 4, 0, map[string]int{"a": 2, "b": 1, "c": 1}, 2.75},
		{5, 5, 0, map[string]int{"a": 2, "b": 1, "c": 2}, 3.125},
		{6, 6, 0, map[string]int{"a": 3, "b": 1, "c": 2}, 3.375},
		{7, 7, 0, map[string]int{"a": 3, "b": 2, "c": 2}, 3.625},
		{10, 7, 3, map[string]int{"a": 3, "b": 2, "c": 2}, 3.625},
	}
	for _, w := range want {
		r, err := e.Select(w.budget)
		if err != nil {
			t.Fatalf("Select(%d): %v", w.budget, err)
		}
		if r.Used != w.used || r.Remaining != w.remaining {
			t.Fatalf("budget %d: used=%d remaining=%d, want %d/%d",
				w.budget, r.Used, r.Remaining, w.used, w.remaining)
		}
		if !floatClose(r.Payoff, w.payoff) {
			t.Fatalf("budget %d: payoff=%.17g, want %.4f", w.budget, r.Payoff, w.payoff)
		}
		sels := selByID(r)
		for id, length := range w.lens {
			if sels[id].Length != length {
				t.Fatalf("budget %d, %q: length=%d, want %d", w.budget, id, sels[id].Length, length)
			}
		}
		// 与穷举参考交叉验证长度与收益。
		refLens, refPayoff := exhaustive(t, reqs, w.budget)
		for i, s := range r.Selections {
			if s.Length != refLens[i] {
				t.Fatalf("budget %d: %q length %d != reference %d",
					w.budget, s.RequestID, s.Length, refLens[i])
			}
		}
		if !floatClose(r.Payoff, ratFloat(refPayoff)) {
			t.Fatalf("budget %d: payoff %.17g != reference %.17g", w.budget, r.Payoff, ratFloat(refPayoff))
		}
	}

	// 抽查真实选择内容（位置、词、单请求累计收益），预算 4。
	r, _ := e.Select(4)
	sels := selByID(r)
	if got := sels["a"]; got.Length != 2 ||
		!reflect.DeepEqual(got.Positions, []int{0, 1}) ||
		!reflect.DeepEqual(got.Words, []string{"a0", "a1"}) ||
		!floatClose(got.Payoff, 1.5) {
		t.Fatalf("unexpected selection for a at budget 4: %+v", got)
	}
	if got := sels["b"]; got.Length != 1 || got.Positions[0] != 0 ||
		got.Words[0] != "b0" || !floatClose(got.Payoff, 0.5) {
		t.Fatalf("unexpected selection for b at budget 4: %+v", got)
	}
	if got := sels["c"]; got.Length != 1 || got.Words[0] != "c0" ||
		!floatClose(got.Payoff, 0.75) {
		t.Fatalf("unexpected selection for c at budget 4: %+v", got)
	}
}

// TestStableTies 验证相同收益按请求标识、再按位置确定稳定顺序。
func TestStableTies(t *testing.T) {
	t.Run("identical chains", func(t *testing.T) {
		reqs := []ds.Request{
			{ID: "y", Words: []string{"y0", "y1"}, Probs: []float64{0.5, 0.5}},
			{ID: "x", Words: []string{"x0", "x1"}, Probs: []float64{0.5, 0.5}},
		}
		e := mustBuild(t, reqs)
		r, _ := e.Select(3)
		sels := selByID(r)
		// 所有边际收益都是 .5/.25：x 优先于 y；每链位置 0 优先于位置 1。
		if sels["x"].Length != 2 || !reflect.DeepEqual(sels["x"].Positions, []int{0, 1}) {
			t.Fatalf("x selection = %+v, want length 2 positions [0 1]", sels["x"])
		}
		if sels["y"].Length != 1 || !reflect.DeepEqual(sels["y"].Positions, []int{0}) {
			t.Fatalf("y selection = %+v, want length 1 position [0]", sels["y"])
		}
		assertResultAgainstReference(t, reqs, 3)
	})

	t.Run("cross-position tie", func(t *testing.T) {
		// b0=1 后，a0=0.5 与 b1=0.5 并列：标识更小的 a 先被选中。
		reqs := []ds.Request{
			{ID: "b", Words: []string{"b0", "b1"}, Probs: []float64{1, 0.5}},
			{ID: "a", Words: []string{"a0"}, Probs: []float64{0.5}},
		}
		r := assertResultAgainstReference(t, reqs, 2)
		sels := selByID(r)
		if sels["a"].Length != 1 || sels["b"].Length != 1 {
			t.Fatalf("lengths a=%d b=%d, want 1/1", sels["a"].Length, sels["b"].Length)
		}
	})

	t.Run("zero marginal ties", func(t *testing.T) {
		reqs := []ds.Request{
			{ID: "q", Words: []string{"q0", "q1"}, Probs: []float64{0, 0}},
			{ID: "p", Words: []string{"p0", "p1"}, Probs: []float64{0, 0}},
		}
		e := mustBuild(t, reqs)
		// 全零并列时，低标识链新暴露的下一位置仍以标识优先，
		// 因此贪心先填满整条 p 链再取 q（与穷举参考的字典序规则一致）。
		check := func(budget, pl, ql int) {
			t.Helper()
			r, _ := e.Select(budget)
			sels := selByID(r)
			if sels["p"].Length != pl || sels["q"].Length != ql {
				t.Fatalf("budget %d: p=%d q=%d, want %d/%d",
					budget, sels["p"].Length, sels["q"].Length, pl, ql)
			}
		}
		check(1, 1, 0) // p0 先于 q0
		check(2, 2, 0) // p1 与 q0 并列时仍由 p 的更小标识胜出
		check(3, 2, 1)
		check(4, 2, 2)
	})
}

// TestInsufficientCandidates 候选不足时全部选择并报告剩余额度。
func TestInsufficientCandidates(t *testing.T) {
	reqs := []ds.Request{
		{ID: "a", Words: []string{"a0"}, Probs: []float64{0.8}},
		{ID: "b", Words: []string{"b0", "b1"}, Probs: []float64{0.5, 0.5}},
	}
	e := mustBuild(t, reqs)
	if e.TotalCandidates() != 3 {
		t.Fatalf("total candidates=%d, want 3", e.TotalCandidates())
	}
	r, err := e.Select(10)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if r.Used != 3 || r.Remaining != 7 {
		t.Fatalf("used=%d remaining=%d, want 3/7", r.Used, r.Remaining)
	}
	sels := selByID(r)
	if sels["a"].Length != 1 || sels["b"].Length != 2 {
		t.Fatalf("lengths a=%d b=%d, want 1/2", sels["a"].Length, sels["b"].Length)
	}
	wantPayoff := 0.8 + 0.5 + 0.25
	if !floatClose(r.Payoff, wantPayoff) {
		t.Fatalf("payoff=%.17g, want %.17g", r.Payoff, wantPayoff)
	}
	// 预算恰好等于候选总数：剩余额度为 0。
	r2, _ := e.Select(3)
	if r2.Used != 3 || r2.Remaining != 0 || !floatClose(r2.Payoff, wantPayoff) {
		t.Fatalf("budget 3: used=%d remaining=%d payoff=%.17g", r2.Used, r2.Remaining, r2.Payoff)
	}
}

// TestEmptyBatchAndZeroBudget 空批次与零预算必须有明确结果。
func TestEmptyBatchAndZeroBudget(t *testing.T) {
	e, err := ds.Build(nil)
	if err != nil {
		t.Fatalf("Build(nil): %v", err)
	}
	if e.NumRequests() != 0 || e.TotalCandidates() != 0 {
		t.Fatalf("empty engine sizes: %d/%d", e.NumRequests(), e.TotalCandidates())
	}
	for _, budget := range []int{0, 4} {
		r, err := e.Select(budget)
		if err != nil {
			t.Fatalf("Select(%d): %v", budget, err)
		}
		if r.Used != 0 || r.Remaining != budget || r.Payoff != 0 {
			t.Fatalf("empty batch budget %d: used=%d remaining=%d payoff=%v",
				budget, r.Used, r.Remaining, r.Payoff)
		}
		if len(r.Selections) != 0 {
			t.Fatalf("empty batch: %d selections, want 0", len(r.Selections))
		}
	}

	// 非空批次但零预算：每个请求都出现且长度为 0，位置/词为空切片。
	withEmpty := []ds.Request{
		{ID: "e", Words: nil, Probs: nil},
		{ID: "f", Words: []string{"f0"}, Probs: []float64{0.9}},
	}
	e2 := mustBuild(t, withEmpty)
	r, _ := e2.Select(0)
	if r.Used != 0 || r.Remaining != 0 || r.Payoff != 0 {
		t.Fatalf("zero budget: used=%d remaining=%d payoff=%v", r.Used, r.Remaining, r.Payoff)
	}
	if len(r.Selections) != 2 {
		t.Fatalf("selections=%d, want 2 (empty-candidate request still reported)", len(r.Selections))
	}
	for _, s := range r.Selections {
		if s.Length != 0 || len(s.Positions) != 0 || len(s.Words) != 0 || s.Payoff != 0 {
			t.Fatalf("zero-budget selection not empty: %+v", s)
		}
	}
}

// TestZeroProbability 中途出现零概率：收益归零，但前缀约束仍然成立。
func TestZeroProbability(t *testing.T) {
	reqs := []ds.Request{
		{ID: "m", Words: []string{"m0", "m1", "m2"}, Probs: []float64{0.5, 0, 0.5}},
	}
	e := mustBuild(t, reqs)
	r2, _ := e.Select(2)
	s2 := selByID(r2)["m"]
	if s2.Length != 2 || !reflect.DeepEqual(s2.Positions, []int{0, 1}) ||
		!reflect.DeepEqual(s2.Words, []string{"m0", "m1"}) || s2.Payoff != 0.5 {
		t.Fatalf("budget 2 selection: %+v", s2)
	}
	r3, _ := e.Select(3)
	s3 := selByID(r3)["m"]
	// 零概率之后的位置收益为 0，但只要预算允许仍会沿前缀被选中，不发生跳跃。
	if s3.Length != 3 || !reflect.DeepEqual(s3.Positions, []int{0, 1, 2}) ||
		!reflect.DeepEqual(s3.Words, []string{"m0", "m1", "m2"}) || s3.Payoff != 0.5 {
		t.Fatalf("budget 3 selection: %+v", s3)
	}

	// 起点即零概率：正收益的另一请求优先；零位置按标识顺序消耗剩余预算。
	mixed := []ds.Request{
		{ID: "z", Words: []string{"z0"}, Probs: []float64{0}},
		{ID: "o", Words: []string{"o0"}, Probs: []float64{0.25}},
	}
	e2 := mustBuild(t, mixed)
	r, _ := e2.Select(1)
	sels := selByID(r)
	if sels["o"].Length != 1 || sels["z"].Length != 0 || r.Payoff != 0.25 {
		t.Fatalf("positive-first selection: o=%d z=%d payoff=%v",
			sels["o"].Length, sels["z"].Length, r.Payoff)
	}
	rAll, _ := e2.Select(2)
	if rAll.Used != 2 || rAll.Remaining != 0 || !floatClose(rAll.Payoff, 0.25) {
		t.Fatalf("all selection: %+v", rAll)
	}
}

// TestLongPrefixUnderflow 极长前缀的极小概率下溢为 0 时，
// 前缀闭合仍由选择结构保证，结果有限、可预测。
func TestLongPrefixUnderflow(t *testing.T) {
	const n = 2000
	long := ds.Request{ID: "long", Words: make([]string, n), Probs: make([]float64, n)}
	for i := range long.Probs {
		long.Probs[i] = 0.5
		long.Words[i] = "w"
	}
	short := ds.Request{ID: "s", Words: []string{"s0"}, Probs: []float64{0.75}}
	reqs := []ds.Request{long, short}

	e := mustBuild(t, reqs)
	total := e.TotalCandidates()
	if total != n+1 {
		t.Fatalf("total candidates=%d, want %d", total, n+1)
	}

	// 用满全部候选：即使尾部概率下溢为 0，也没有位置被跳过。
	r, err := e.Select(total)
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if r.Used != total || r.Remaining != 0 {
		t.Fatalf("used=%d remaining=%d, want %d/0", r.Used, r.Remaining, total)
	}
	sels := selByID(r)
	if sels["long"].Length != n {
		t.Fatalf("long length=%d, want %d (underflow must not break prefix closure)", sels["long"].Length, n)
	}
	if !reflect.DeepEqual(sels["long"].Positions[:4], []int{0, 1, 2, 3}) {
		t.Fatalf("long positions prefix broken: %v", sels["long"].Positions[:4])
	}
	if sels["s"].Length != 1 {
		t.Fatalf("short length=%d, want 1", sels["s"].Length)
	}
	// 长链收益夹在 1（首位置）与几何级数上界 2 之间。
	if sels["long"].Payoff < 1 || sels["long"].Payoff > 2 {
		t.Fatalf("long payoff=%.17g outside [1,2]", sels["long"].Payoff)
	}
	if math.IsNaN(r.Payoff) || math.IsInf(r.Payoff, 0) {
		t.Fatalf("total payoff non-finite: %v", r.Payoff)
	}

	// 小预算下与穷举参考逐位一致（下溢之前部分）。
	for _, b := range []int{0, 1, 2, 3, 5} {
		rr, err := e.Select(b)
		if err != nil {
			t.Fatalf("Select(%d): %v", b, err)
		}
		refLens, refPayoff := exhaustive(t, reqs, b)
		for i, s := range rr.Selections {
			if s.Length != refLens[i] {
				t.Fatalf("budget %d: %q length=%d, reference=%d", b, s.RequestID, s.Length, refLens[i])
			}
		}
		if !floatClose(rr.Payoff, ratFloat(refPayoff)) {
			t.Fatalf("budget %d: payoff=%.17g reference=%.17g", b, rr.Payoff, ratFloat(refPayoff))
		}
	}
}

// TestOrderIndependence 输入请求顺序变化不能改变结果。
func TestOrderIndependence(t *testing.T) {
	base := []ds.Request{
		{ID: "a", Words: []string{"a0", "a1"}, Probs: []float64{0.5, 0.5}},
		{ID: "b", Words: []string{"b0", "b1"}, Probs: []float64{1, 0.25}},
		{ID: "c", Words: []string{"c0"}, Probs: []float64{0.75}},
	}
	shuffled := []ds.Request{base[2], base[0], base[1]}

	e1 := mustBuild(t, base)
	e2 := mustBuild(t, shuffled)
	for _, b := range []int{0, 1, 2, 3, 4, 5, 6} {
		r1, err := e1.Select(b)
		if err != nil {
			t.Fatalf("Select(%d): %v", b, err)
		}
		r2, err := e2.Select(b)
		if err != nil {
			t.Fatalf("shuffled Select(%d): %v", b, err)
		}
		if !reflect.DeepEqual(r1, r2) {
			t.Fatalf("budget %d: results differ under input reordering:\n%+v\n%+v", b, r1, r2)
		}
	}
}

// TestRandomAgainstExhaustive 随机小规模批次上，将引擎结果与穷举参考
// 逐请求长度、总收益对比，并验证前缀闭合与预算守恒。
func TestRandomAgainstExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(20260930))
	probChoices := []float64{0, 0.125, 0.25, 0.5, 0.75, 1}
	for trial := 0; trial < 400; trial++ {
		n := rng.Intn(5)
		reqs := make([]ds.Request, n)
		usedIDs := map[string]bool{}
		total := 0
		for i := 0; i < n; i++ {
			id := string(rune('r')) + itoa(i) + "_" + itoa(trial)
			for usedIDs[id] {
				id += "x"
			}
			usedIDs[id] = true
			l := rng.Intn(6)
			total += l
			req := ds.Request{ID: id, Words: make([]string, l), Probs: make([]float64, l)}
			for k := 0; k < l; k++ {
				req.Probs[k] = probChoices[rng.Intn(len(probChoices))]
				req.Words[k] = id + ":w" + itoa(k)
			}
			reqs[i] = req
		}

		budgets := []int{0, total}
		if total > 0 {
			budgets = append(budgets, rng.Intn(total+1))
		}
		budgets = append(budgets, total+rng.Intn(4))
		for _, b := range budgets {
			assertResultAgainstReference(t, reqs, b)
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// TestValidationErrors 覆盖重复标识、负预算、长度不符、
// 越界及非有限概率等全部拒绝路径。
func TestValidationErrors(t *testing.T) {
	if _, err := ds.Build([]ds.Request{
		{ID: "a"}, {ID: "b"}, {ID: "a"},
	}); !errors.Is(err, ds.ErrDuplicateID) {
		t.Fatalf("duplicate id: err=%v, want ErrDuplicateID", err)
	}

	if _, err := ds.Build([]ds.Request{
		{ID: "a", Words: []string{"w0", "w1"}, Probs: []float64{0.5}},
	}); !errors.Is(err, ds.ErrLengthMismatch) {
		t.Fatalf("length mismatch: err=%v, want ErrLengthMismatch", err)
	}

	badProbs := []float64{-0.0001, 1.0001, math.NaN(), math.Inf(1), math.Inf(-1)}
	for i, p := range badProbs {
		probs := []float64{0.5, 0.5, p} // 越界/非有限出现在靠后位置也必须被发现
		if _, err := ds.Build([]ds.Request{
			{ID: "a", Words: []string{"w0", "w1", "w2"}, Probs: probs},
		}); !errors.Is(err, ds.ErrInvalidProbability) {
			t.Fatalf("bad probs case %d (p=%v): err=%v, want ErrInvalidProbability", i, p, err)
		}
	}

	e := mustBuild(t, []ds.Request{{ID: "a", Words: []string{"a0"}, Probs: []float64{0.5}}})
	if _, err := e.Select(-1); !errors.Is(err, ds.ErrNegativeBudget) {
		t.Fatalf("negative budget: err=%v, want ErrNegativeBudget", err)
	}
	if _, err := e.SelectMany([]int{0, -3}); !errors.Is(err, ds.ErrNegativeBudget) {
		t.Fatalf("SelectMany negative budget: err=%v, want ErrNegativeBudget", err)
	}
}

// TestSelectMany 同一输入计算多个预算，结果与逐个 Select 完全一致。
func TestSelectMany(t *testing.T) {
	reqs := []ds.Request{
		{ID: "a", Words: []string{"a0", "a1", "a2"}, Probs: []float64{1, 0.5, 0.5}},
		{ID: "b", Words: []string{"b0", "b1"}, Probs: []float64{0.5, 0.5}},
		{ID: "c", Words: []string{"c0", "c1"}, Probs: []float64{0.75, 0.5}},
	}
	e := mustBuild(t, reqs)
	budgets := []int{0, 2, 5, 7, 100}
	results, err := e.SelectMany(budgets)
	if err != nil {
		t.Fatalf("SelectMany: %v", err)
	}
	if len(results) != len(budgets) {
		t.Fatalf("got %d results, want %d", len(results), len(budgets))
	}
	for i, b := range budgets {
		one, err := e.Select(b)
		if err != nil {
			t.Fatalf("Select(%d): %v", b, err)
		}
		if !reflect.DeepEqual(results[i], one) {
			t.Fatalf("SelectMany[%d] differs from Select(%d):\n%+v\n%+v", i, b, results[i], one)
		}
		if results[i].Budget != b {
			t.Fatalf("result budget=%d, want %d", results[i].Budget, b)
		}
	}

	empty, err := e.SelectMany(nil)
	if err != nil || len(empty) != 0 {
		t.Fatalf("SelectMany(nil) = (%v, %v)", empty, err)
	}
}
