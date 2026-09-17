/** Fit a grayscale 28x28 drawing to 20x20 and center its intensity-weighted mass. */
export function normalizeDigit(pixels) {
    if (pixels.length !== 784 || pixels.some(value => !Number.isFinite(value) || value < 0 || value > 1)) {
        throw Error("Expected 784 grayscale values between 0 and 1.");
    }
    let minX = 28, minY = 28, maxX = -1, maxY = -1;
    let mass = 0, momentX = 0, momentY = 0;
    for (let i = 0; i < pixels.length; i++) {
        const value = pixels[i], x = i % 28, y = Math.floor(i / 28);
        if (value > 0) {
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            mass += value; momentX += x * value; momentY += y * value;
        }
    }
    if (!mass) return Array(784).fill(0);
    const cx = momentX / mass, cy = momentY / mass;
    // Leave room for highly asymmetric shapes when centering by mass.
    const extent = Math.max(cx - minX + .5, maxX - cx + .5, cy - minY + .5, maxY - cy + .5);
    const scale = Math.min(20 / Math.max(maxX - minX + 1, maxY - minY + 1), 13 / extent);
    const pixel = (x, y) => x < 0 || x >= 28 || y < 0 || y >= 28 ? 0 : pixels[y * 28 + x];
    return Array.from({length: 784}, (_, i) => {
        const x = (i % 28 - 13.5) / scale + cx;
        const y = (Math.floor(i / 28) - 13.5) / scale + cy;
        const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
        const value = (1 - fy) * ((1 - fx) * pixel(x0, y0) + fx * pixel(x0 + 1, y0))
            + fy * ((1 - fx) * pixel(x0, y0 + 1) + fx * pixel(x0 + 1, y0 + 1));
        return Math.max(0, Math.min(1, value));
    });
}
