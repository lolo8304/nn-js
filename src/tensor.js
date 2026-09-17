function sizeOf(s) {
    return s.reduce((a, b) => a * b, 1)
}

function stridesOf(s) {
    const r = new Array(s.length);
    let n = 1;
    for (let i = s.length - 1; i >= 0; i--) {
        r[i] = n;
        n *= s[i]
    }
    return r
}

function each(s, f) {
    if (!s.length) {
        f([]);
        return
    }
    if (s.some(x => x === 0)) return;
    const i = new Array(s.length).fill(0);
    for (; ;) {
        f(i.slice());
        let a = s.length - 1;
        while (a >= 0 && ++i[a] === s[a]) {
            i[a] = 0;
            a--
        }
        if (a < 0) return
    }
}

function bshape(a, b) {
    const n = Math.max(a.length, b.length), o = new Array(n);
    for (let i = 0; i < n; i++) {
        const x = a[a.length - 1 - i] ?? 1, y = b[b.length - 1 - i] ?? 1;
        if (x !== y && x !== 1 && y !== 1) throw Error(`Cannot broadcast [${a}] and [${b}]`);
        o[n - 1 - i] = Math.max(x, y)
    }
    return o
}

export class Tensor {
    constructor(data, shape, strides = stridesOf(shape), offset = 0) {
        this.data = data instanceof Float64Array ? data : Float64Array.from(data);
        this.shape = [...shape];
        this.strides = [...strides];
        this.offset = offset
    }

    get rank() {
        return this.shape.length
    }

    get size() {
        return sizeOf(this.shape)
    }

    static zeros(...s) {
        return new Tensor(new Float64Array(sizeOf(s)), s)
    }

    static scalar(x) {
        return new Tensor(Float64Array.of(x), [])
    }

    static of(v, shape = null) {
        if (shape) return new Tensor(v, shape);
        if (Array.isArray(v) && Array.isArray(v[0])) {
            const r = v.length, c = r ? v[0].length : 0;
            if (v.some(x => x.length !== c)) throw Error("Ragged array");
            return new Tensor(Float64Array.from(v.flat()), [r, c])
        }
        return new Tensor(Float64Array.from(v), [v.length])
    }

    axis(a) {
        if (a < 0) a += this.rank;
        if (a < 0 || a >= this.rank) throw Error("Bad axis");
        return a
    }

    address(i) {
        if (i.length !== this.rank) throw Error(`Expected ${this.rank} indices`);
        let p = this.offset;
        for (let a = 0; a < i.length; a++) {
            if (i[a] < 0 || i[a] >= this.shape[a]) throw RangeError("Tensor index");
            p += i[a] * this.strides[a]
        }
        return p
    }

    get(...i) {
        return this.data[this.address(i)]
    }

    set(v, ...i) {
        this.data[this.address(i)] = v;
        return this
    }

    toArray() {
        const o = new Float64Array(this.size);
        let p = 0;
        each(this.shape, i => o[p++] = this.get(...i));
        return o
    }

    slice(a, n) {
        a = this.axis(a);
        const s = this.shape.filter((_, i) => i !== a), st = this.strides.filter((_, i) => i !== a);
        return new Tensor(this.data, s, st, this.offset + n * this.strides[a])
    }

    reshape(...s) {
        const neg = s.indexOf(-1);
        if (neg >= 0) {
            if (s.lastIndexOf(-1) !== neg) throw Error("Only one -1");
            const k = s.reduce((p, x) => x === -1 ? p : p * x, 1);
            if (this.size % k) throw Error("Cannot infer");
            s[neg] = this.size / k
        }
        if (sizeOf(s) !== this.size) throw Error("Size mismatch");
        return new Tensor(this.toArray(), s)
    }

    flatten() {
        return this.reshape(this.size)
    }

    map(f) {
        const o = Tensor.zeros(...this.shape);
        each(this.shape, i => o.set(f(this.get(...i)), ...i));
        return o
    }

    bget(oi) {
        const sh = oi.length - this.rank, i = this.shape.map((s, a) => s === 1 ? 0 : oi[a + sh]);
        return this.get(...i)
    }

    binary(b, f) {
        if (!(b instanceof Tensor)) b = Tensor.scalar(b);
        const s = bshape(this.shape, b.shape), o = Tensor.zeros(...s);
        each(s, i => o.set(f(this.bget(i), b.bget(i)), ...i));
        return o
    }

    add(b) {
        return this.binary(b, (x, y) => x + y)
    }

    matmul(b) {
        if (!this.rank || !b.rank) throw Error("Scalar matmul");
        const av = this.rank === 1, bv = b.rank === 1, m = av ? 1 : this.shape.at(-2), k = this.shape.at(-1),
            bk = bv ? b.shape[0] : b.shape.at(-2), n = bv ? 1 : b.shape.at(-1);
        if (k !== bk) throw Error("Inner dimensions differ");
        const ab = av ? [] : this.shape.slice(0, -2), bb = bv ? [] : b.shape.slice(0, -2), batch = bshape(ab, bb);
        if (av && bv) {
            let s = 0;
            for (let q = 0; q < k; q++) s += this.get(q) * b.get(q);
            return Tensor.scalar(s)
        }
        const os = [...batch, ...(av ? [] : [m]), ...(bv ? [] : [n])], o = Tensor.zeros(...os);
        const mg = (t, bi, row, col, vec) => {
            if (vec) return t.get(col);
            const br = t.rank - 2, shift = bi.length - br, idx = new Array(t.rank);
            for (let x = 0; x < br; x++) idx[x] = t.shape[x] === 1 ? 0 : bi[x + shift];
            idx[t.rank - 2] = row;
            idx[t.rank - 1] = col;
            return t.get(...idx)
        };
        each(os, oi => {
            const bi = oi.slice(0, batch.length), row = av ? 0 : oi[batch.length], col = bv ? 0 : oi.at(-1);
            let s = 0;
            for (let q = 0; q < k; q++) s += mg(this, bi, row, q, av) * mg(b, bi, q, col, bv);
            o.set(s, ...oi)
        });
        return o
    }

    relu() {
        return this.map(x => Math.max(0, x))
    }

    sigmoid() {
        return this.map(x => 1 / (1 + Math.exp(-x)))
    }

    tanh() {
        return this.map(Math.tanh)
    }

    leakyRelu(a = .01) {
        return this.map(x => x >= 0 ? x : a * x)
    }

    softmax(a = -1) {
        a = this.axis(a);
        const o = Tensor.zeros(...this.shape), outer = this.shape.filter((_, i) => i !== a);
        each(outer, z => {
            const ix = k => {
                const x = z.slice();
                x.splice(a, 0, k);
                return x
            };
            let m = -Infinity;
            for (let k = 0; k < this.shape[a]; k++) m = Math.max(m, this.get(...ix(k)));
            let sum = 0;
            for (let k = 0; k < this.shape[a]; k++) {
                const i = ix(k), e = Math.exp(this.get(...i) - m);
                o.set(e, ...i);
                sum += e
            }
            for (let k = 0; k < this.shape[a]; k++) {
                const i = ix(k);
                o.set(o.get(...i) / sum, ...i)
            }
        });
        return o
    }

    toNestedArray() {
        const rec = (d, p) => d === this.rank ? this.get(...p) : Array.from({length: this.shape[d]}, (_, i) => rec(d + 1, [...p, i]));
        return rec(0, [])
    }
}
