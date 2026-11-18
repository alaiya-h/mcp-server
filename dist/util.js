import fs from "node:fs";
import path from "node:path";
/* ------------------------------------------------------------------ */
/* Repo root + sandbox                                                  */
/* ------------------------------------------------------------------ */
export const MAX_FILES = 50_000;
export const IGNORED_DIRS = new Set([
    "node_modules",
    ".git",
    ".next",
    ".vercel",
    ".turbo",
    ".cache",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".venv",
    "venv",
]);
export const SRC_EXTS = new Set([
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".html", ".vue", ".svelte",
]);
/** The repo being inspected. Set REPO_ROOT or pass the path as the first CLI argument. */
export const REPO_ROOT = (() => {
    const raw = process.env.REPO_ROOT ?? process.argv[2] ?? process.cwd();
    const abs = path.resolve(raw);
    if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) {
        console.error(`REPO_ROOT does not exist or is not a directory: ${abs}`);
        process.exit(1);
    }
    return fs.realpathSync(abs);
})();
function isInside(abs) {
    const rel = path.relative(REPO_ROOT, abs);
    return !(rel.startsWith("..") || path.isAbsolute(rel));
}
/** Resolve a repo-relative path and refuse anything that escapes the repo (including via symlinks). */
export function resolveInRepo(rel = ".") {
    const abs = path.resolve(REPO_ROOT, rel);
    if (!isInside(abs))
        throw new Error(`Path escapes the repository: ${rel}`);
    if (fs.existsSync(abs) && !isInside(fs.realpathSync(abs))) {
        throw new Error(`Path resolves (via symlink) outside the repository: ${rel}`);
    }
    return abs;
}
export function toRel(abs) {
    return path.relative(REPO_ROOT, abs).split(path.sep).join("/");
}
export function fmtBytes(n) {
    if (n < 0)
        return "unknown";
    const units = ["B", "KB", "MB", "GB"];
    let v = n;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
        v /= 1024;
        i++;
    }
    return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
}
/** Depth-first walk. maxDepth=1 lists only immediate children. Symlinks are never followed. */
export function* walk(start = REPO_ROOT, maxDepth = Infinity) {
    const state = { count: 0 };
    function* rec(dir, depth) {
        let items;
        try {
            items = fs.readdirSync(dir, { withFileTypes: true });
        }
        catch {
            return;
        }
        items.sort((a, b) => a.name.localeCompare(b.name));
        for (const it of items) {
            if (it.isSymbolicLink())
                continue;
            const abs = path.join(dir, it.name);
            if (it.isDirectory()) {
                if (IGNORED_DIRS.has(it.name))
                    continue;
                yield { abs, rel: toRel(abs), isDir: true, size: 0 };
                if (depth < maxDepth)
                    yield* rec(abs, depth + 1);
            }
            else if (it.isFile()) {
                if (++state.count > MAX_FILES)
                    return;
                let size = 0;
                try {
                    size = fs.statSync(abs).size;
                }
                catch {
                    /* ignore */
                }
                yield { abs, rel: toRel(abs), isDir: false, size };
            }
        }
    }
    yield* rec(start, 1);
}
export function allFiles() {
    return [...walk()].filter((e) => !e.isDir);
}
export function sourceFiles() {
    return allFiles().filter((e) => SRC_EXTS.has(path.extname(e.rel).toLowerCase()) && e.size < 1_000_000);
}
/* ------------------------------------------------------------------ */
/* Reading                                                              */
/* ------------------------------------------------------------------ */
export function readHead(abs, n = 512) {
    const fd = fs.openSync(abs, "r");
    try {
        const buf = Buffer.alloc(n);
        const read = fs.readSync(fd, buf, 0, n, 0);
        return buf.subarray(0, read);
    }
    finally {
        fs.closeSync(fd);
    }
}
export function isBinary(buf) {
    return buf.includes(0);
}
export function readTextRange(abs, startLine = 1, endLine, maxChars = 200_000) {
    const st = fs.statSync(abs);
    if (!st.isFile())
        throw new Error("Not a file.");
    if (isBinary(readHead(abs, 8192))) {
        throw new Error("Binary file. Use inspect_onnx_model, inspect_git_lfs or find_model_files for model files.");
    }
    if (st.size > 5_000_000)
        throw new Error(`File too large to read as text (${fmtBytes(st.size)}).`);
    const lines = fs.readFileSync(abs, "utf8").split(/\r?\n/);
    const from = Math.max(1, startLine);
    const to = Math.min(lines.length, endLine ?? lines.length);
    let content = "";
    let truncated = false;
    for (let i = from; i <= to; i++) {
        const l = `${i}: ${lines[i - 1]}\n`;
        if (content.length + l.length > maxChars) {
            truncated = true;
            break;
        }
        content += l;
    }
    return { totalLines: lines.length, from, to, truncated, content };
}
/** A Git LFS pointer is a ~130 byte text file starting with the spec line. */
export function parseLfsPointer(head) {
    const text = head.toString("utf8");
    if (!text.startsWith("version https://git-lfs.github.com/spec/v1"))
        return null;
    const oid = /oid sha256:([0-9a-f]{64})/.exec(text)?.[1] ?? "unknown";
    const size = /size (\d+)/.exec(text)?.[1];
    return { oid, size: size ? Number(size) : -1 };
}
/* ------------------------------------------------------------------ */
/* Globs + grep                                                         */
/* ------------------------------------------------------------------ */
/** Minimal gitattributes/gitignore-style glob -> RegExp (supports *, **, ?). */
export function globToRegExp(glob) {
    const trimmed = glob.trim();
    const g = trimmed.replace(/^\//, "");
    const anchored = trimmed.startsWith("/") || g.includes("/");
    let re = "";
    for (let i = 0; i < g.length; i++) {
        const c = g[i];
        if (c === "*") {
            if (g[i + 1] === "*") {
                re += ".*";
                i++;
                if (g[i + 1] === "/")
                    i++;
            }
            else {
                re += "[^/]*";
            }
        }
        else if (c === "?") {
            re += "[^/]";
        }
        else {
            re += c.replace(/[.+^${}()|[\]\\]/g, "\\$&");
        }
    }
    return new RegExp(anchored ? `^${re}$` : `(^|/)${re}$`);
}
/** Line-by-line search. The regex must NOT have the g flag. */
export function grepFiles(files, re, maxHits = 50) {
    const hits = [];
    for (const f of files) {
        if (f.size > 1_000_000)
            continue;
        let text;
        try {
            text = fs.readFileSync(f.abs, "utf8");
        }
        catch {
            continue;
        }
        if (text.includes("\u0000"))
            continue;
        const lines = text.split(/\r?\n/);
        for (let i = 0; i < lines.length; i++) {
            if (re.test(lines[i])) {
                hits.push({ file: f.rel, line: i + 1, text: lines[i].trim().slice(0, 240) });
                if (hits.length >= maxHits)
                    return hits;
            }
        }
    }
    return hits;
}
export function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
