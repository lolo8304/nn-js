import {Tensor} from "./tensor.js";

const MAGIC = 0x4A4E4E31;

class Reader {
    constructor(buffer) {
        this.v = new DataView(buffer);
        this.p = 0
    }

    int() {
        const x = this.v.getInt32(this.p, false);
        this.p += 4;
        return x
    }

    double() {
        const x = this.v.getFloat64(this.p, false);
        this.p += 8;
        return x
    }

    utf() {
        const n = this.v.getUint16(this.p, false);
        this.p += 2;
        const b = new Uint8Array(this.v.buffer, this.v.byteOffset + this.p, n);
        this.p += n;
        return new TextDecoder().decode(b)
    }

    tensor() {
        const r = this.int(), s = new Array(r);
        let n = 1;
        for (let i = 0; i < r; i++) {
            s[i] = this.int();
            n *= s[i]
        }
        const a = new Float64Array(n);
        for (let i = 0; i < n; i++) a[i] = this.double();
        return new Tensor(a, s)
    }
}

export class SequentialModel {
    constructor(layers) {
        this.layers = layers
    }

    static fromArrayBuffer(buffer) {
        if (ArrayBuffer.isView(buffer)) buffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
        const r = new Reader(buffer);
        if (r.int() !== MAGIC) throw Error("Not a JNN1 .nn model");
        const version = r.int();
        if (version !== 1) throw Error(`Unsupported JNN version ${version}`);
        const n = r.int(), ls = [];
        for (let i = 0; i < n; i++) {
            const type = r.utf();
            switch (type) {
                case"Dense":
                    ls.push({type, inputSize: r.int(), outputSize: r.int(), weight: r.tensor(), bias: r.tensor()});
                    break;
                case"LeakyReLU":
                    ls.push({type, alpha: r.double()});
                    break;
                case"Softmax":
                    ls.push({type, axis: r.int()});
                    break;
                case"Dropout":
                    ls.push({type, probability: r.double()});
                    break;
                case"ReLU":
                case"Sigmoid":
                case"Tanh":
                case"Flatten":
                    ls.push({type});
                    break;
                default:
                    throw Error(`Unsupported layer ${type}`)
            }
        }
        return new SequentialModel(ls)
    }

    static async load(source) {
        if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) return this.fromArrayBuffer(source);
        if (typeof Blob !== "undefined" && source instanceof Blob) return this.fromArrayBuffer(await source.arrayBuffer());
        if (typeof source === "string" || source instanceof URL) {
            const res = await fetch(source);
            if (!res.ok) throw Error(`Could not load model: HTTP ${res.status}`);
            return this.fromArrayBuffer(await res.arrayBuffer())
        }
        throw TypeError("Expected File/Blob, ArrayBuffer, TypedArray, or URL")
    }

    predict(input) {
        let x = input instanceof Tensor ? input : Tensor.of(input);
        for (const l of this.layers) switch (l.type) {
            case"Dense":
                x = x.matmul(l.weight).add(l.bias);
                break;
            case"ReLU":
                x = x.relu();
                break;
            case"Sigmoid":
                x = x.sigmoid();
                break;
            case"Tanh":
                x = x.tanh();
                break;
            case"LeakyReLU":
                x = x.leakyRelu(l.alpha);
                break;
            case"Softmax":
                x = x.softmax(l.axis);
                break;
            case"Flatten":
                if (x.rank > 1) {
                    const batch = x.shape[0], rest = x.shape.slice(1).reduce((a, b) => a * b, 1);
                    x = x.reshape(batch, rest)
                }
                break;
            case"Dropout":
                break
        }
        return x
    }

    predictClasses(input) {
        const p = this.predict(input);
        if (p.rank !== 2) throw Error("Expected [batch, classes]");
        const o = Tensor.zeros(p.shape[0]);
        for (let i = 0; i < p.shape[0]; i++) {
            let best = 0;
            for (let j = 1; j < p.shape[1]; j++) if (p.get(i, j) > p.get(i, best)) best = j;
            o.set(best, i)
        }
        return o
    }

    summary() {
        return this.layers.map((l, i) => `${i}: ${l.type}${l.type === "Dense" ? ` (${l.inputSize} -> ${l.outputSize})` : ""}`).join("\n")
    }
}
