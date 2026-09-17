import test from "node:test";
import assert from "node:assert/strict";
import {normalizeDigit} from "../examples/browser/normalize-digit.js";

function rectangle(left, top, width, height) {
    return Array.from({length: 784}, (_, i) => {
        const x = i % 28, y = Math.floor(i / 28);
        return x >= left && x < left + width && y >= top && y < top + height ? 1 : 0;
    });
}

test("blank input stays blank and invalid inputs are rejected", () => {
    assert.deepEqual(normalizeDigit(Array(784).fill(0)), Array(784).fill(0));
    for (const input of [[], Array(784).fill(NaN), Array(784).fill(-1), Array(784).fill(2)]) {
        assert.throws(() => normalizeDigit(input));
    }
});

test("normalization removes translation without mutating the drawing", () => {
    const original = rectangle(0, 0, 4, 10), before = [...original];
    const result = normalizeDigit(original);
    assert.deepEqual(result, normalizeDigit(rectangle(22, 16, 4, 10)));
    assert.deepEqual(original, before);
    assert.equal(result.length, 784);
    assert.ok(result.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
    const mass = result.reduce((a, b) => a + b, 0);
    const x = result.reduce((sum, value, i) => sum + value * (i % 28), 0) / mass;
    const y = result.reduce((sum, value, i) => sum + value * Math.floor(i / 28), 0) / mass;
    assert.ok(Math.abs(x - 13.5) < .01);
    assert.ok(Math.abs(y - 13.5) < .01);
});

test("different drawing scales fit the same 20-pixel extent", () => {
    for (const pixels of [rectangle(3, 4, 4, 10), rectangle(9, 2, 8, 20)]) {
        const result = normalizeDigit(pixels);
        const ys = result.flatMap((v, i) => v >= .5 ? [Math.floor(i / 28)] : []);
        assert.equal(Math.max(...ys) - Math.min(...ys) + 1, 20);
    }
});
