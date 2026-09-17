import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const browserRoot = path.join(root, "examples/browser");
const sourceRoot = path.join(root, "src");
const types = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".nn": "application/octet-stream"
};
const port = Number(process.env.PORT || 8080);

http.createServer((req, res) => {
    let pathname;
    try {
        pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
        if (pathname.includes("\0")) throw new Error("Invalid path");
    } catch {
        res.writeHead(400).end("Bad request");
        return;
    }
    // Serve browser assets at / and expose the library modules at /src/.
    const base = pathname.startsWith("/src/") ? sourceRoot : browserRoot;
    if (base === sourceRoot) {
        pathname = pathname.slice("/src".length);
    } else if (pathname.startsWith("/examples/browser/")) {
        pathname = pathname.slice("/examples/browser".length);
    }
    if (pathname === "/") pathname = "/index.html";
    const file = path.resolve(base, `.${pathname}`);
    const relative = path.relative(base, file);
    if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
        res.writeHead(403).end();
        return;
    }
    fs.readFile(file, (error, data) => {
        if (error) {
            res.writeHead(error.code === "ENOENT" || error.code === "EISDIR" ? 404 : 500).end("Could not load file");
            return;
        }
        res.setHeader("Content-Type", types[path.extname(file)] || "application/octet-stream");
        res.end(data);
    });
}).listen(port, function () {
    console.log(`Open http://localhost:${this.address().port}`);
});
