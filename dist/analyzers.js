import fs from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { REPO_ROOT, allFiles, sourceFiles, resolveInRepo, readHead, parseLfsPointer, fmtBytes, grepFiles, globToRegExp, } from "./util.js";
import { parseOnnx } from "./onnx.js";
const execFileP = promisify(execFile);
function lineOf(text, index) {
    let n = 1;
    for (let i = 0; i < index; i++)
        if (text.charCodeAt(i) === 10)
            n++;
    return n;
}
function readText(rel) {
    const p = path.join(REPO_ROOT, rel);
    try {
        return fs.existsSync(p) ? fs.readFileSync(p, "utf8") : null;
    }
    catch {
        return null;
    }
}
/* ------------------------------------------------------------------ */
/* 1. Model files                                                       */
/* ------------------------------------------------------------------ */
export const MODEL_EXTS = new Set([
    ".onnx", ".ort", ".tflite", ".pt", ".pth", ".h5", ".keras", ".pb",
    ".safetensors", ".pkl", ".joblib", ".mlmodel", ".engine",
]);
export function findModelFiles() {
    const all = allFiles();
    const models = all
        .filter((e) => MODEL_EXTS.has(path.extname(e.rel).toLowerCase()))
        .map((e) => {
        const ptr = e.size <= 1024 ? parseLfsPointer(readHead(e.abs, 1024)) : null;
        return {
            path: e.rel,
            sizeBytes: e.size,
            size: fmtBytes(e.size),
            lfsPointer: ptr ? { oid: ptr.oid, realSize: fmtBytes(ptr.size) } : null,
            status: ptr ? "LFS POINTER (real model not checked out)" : "binary present",
        };
    });
    const labelFiles = all
        .filter((e) => /(label|class|species|taxonom|names)/i.test(path.posix.basename(e.rel)) &&
        /\.(json|txt|csv)$/i.test(e.rel))
        .map((e) => ({ path: e.rel, size: fmtBytes(e.size) }));
    return { models, labelFiles };
}
/* ------------------------------------------------------------------ */
/* 2. ONNX inspection                                                   */
/* ------------------------------------------------------------------ */
export function inspectOnnx(rel) {
    let target = rel;
    if (!target) {
        const m = findModelFiles().models.find((x) => /\.(onnx|ort)$/i.test(x.path));
        if (!m)
            throw new Error("No .onnx/.ort file found in the repo (try find_model_files).");
        target = m.path;
    }
    const abs = resolveInRepo(target);
    if (!fs.existsSync(abs) || !fs.statSync(abs).isFile())
        throw new Error(`File not found: ${target}`);
    const sizeBytes = fs.statSync(abs).size;
    const head = readHead(abs, 1024);
    const ptr = parseLfsPointer(head);
    if (ptr) {
        return {
            path: target,
            valid: false,
            problem: "This file is a Git LFS pointer, not a model. Anything that loads it (onnxruntime, fetch, fs) will fail " +
                "with errors like 'Protobuf parsing failed'.",
            lfsPointer: { oid: ptr.oid, realSize: fmtBytes(ptr.size) },
            fix: "Run 'git lfs pull' locally, and make sure Git LFS is enabled for the Vercel project, then redeploy.",
        };
    }
    const headText = head.toString("utf8", 0, 64).trimStart().toLowerCase();
    if (headText.startsWith("<!doctype html") || headText.startsWith("<html")) {
        return {
            path: target,
            valid: false,
            problem: "File contains HTML (probably a saved 404/login page), not a model.",
        };
    }
    if (sizeBytes === 0)
        return { path: target, valid: false, problem: "File is empty (0 bytes)." };
    if (/\.ort$/i.test(target)) {
        return {
            path: target,
            valid: true,
            size: fmtBytes(sizeBytes),
            note: "ORT-format (FlatBuffers) model; not parsed by this tool.",
        };
    }
    try {
        const summary = parseOnnx(abs);
        return { path: target, valid: true, size: fmtBytes(sizeBytes), summary };
    }
    catch (e) {
        return {
            path: target,
            valid: false,
            size: fmtBytes(sizeBytes),
            problem: `Could not parse as an ONNX protobuf: ${e.message}`,
            hint: "Usually a truncated download/upload, an LFS pointer, an HTML error page, or a non-ONNX file with a .onnx extension.",
        };
    }
}
/* ------------------------------------------------------------------ */
/* 3. Git LFS                                                           */
/* ------------------------------------------------------------------ */
export async function inspectLfs() {
    const findings = [];
    const files = allFiles();
    const rules = [];
    const ga = readText(".gitattributes");
    if (ga) {
        ga.split(/\r?\n/).forEach((t, i) => {
            const s = t.trim();
            if (!s || s.startsWith("#"))
                return;
            if (/\bfilter=lfs\b/.test(s))
                rules.push({ line: i + 1, pattern: s.split(/\s+/)[0], text: s });
        });
    }
    const regs = rules.map((r) => ({ r, re: globToRegExp(r.pattern) }));
    const pointers = [];
    const trackedReal = [];
    const largeUntracked = [];
    for (const f of files) {
        const tracked = regs.some((x) => x.re.test(f.rel));
        const ptr = f.size <= 1024 ? parseLfsPointer(readHead(f.abs, 1024)) : null;
        if (ptr) {
            pointers.push({ path: f.rel, oid: ptr.oid, realSize: fmtBytes(ptr.size), matchesLfsRule: tracked });
        }
        else if (tracked) {
            trackedReal.push({ path: f.rel, size: fmtBytes(f.size) });
        }
        else if (f.size > 50 * 1024 * 1024) {
            largeUntracked.push({ path: f.rel, size: fmtBytes(f.size) });
        }
    }
    const gitLfs = {
        installed: false,
        version: "",
        lsFiles: [],
    };
    try {
        const v = await execFileP("git", ["lfs", "version"], { cwd: REPO_ROOT, timeout: 10_000 });
        gitLfs.installed = true;
        gitLfs.version = v.stdout.trim();
        const ls = await execFileP("git", ["lfs", "ls-files"], { cwd: REPO_ROOT, timeout: 20_000 });
        gitLfs.lsFiles = ls.stdout.split("\n").filter(Boolean).slice(0, 100);
    }
    catch (e) {
        gitLfs.error = e.message.split("\n")[0];
    }
    for (const p of pointers.slice(0, 10)) {
        findings.push({
            severity: "error",
            area: "lfs",
            message: `${p.path} is a Git LFS pointer file (real size ${p.realSize}), not the actual content.`,
            evidence: p,
        });
    }
    if (pointers.some((p) => !p.matchesLfsRule)) {
        findings.push({
            severity: "warn",
            area: "lfs",
            message: "Some LFS pointer files are not covered by a filter=lfs rule in the root .gitattributes.",
        });
    }
    if (rules.length === 0 && pointers.length === 0) {
        findings.push({ severity: "info", area: "lfs", message: "No filter=lfs rules found in the root .gitattributes." });
    }
    if (rules.length > 0 && !gitLfs.installed) {
        findings.push({
            severity: "warn",
            area: "lfs",
            message: "Repo uses Git LFS but 'git lfs' is not available to this server, so ls-files could not be checked.",
            evidence: gitLfs.error,
        });
    }
    for (const l of largeUntracked.slice(0, 5)) {
        findings.push({
            severity: "warn",
            area: "lfs",
            message: `${l.path} is ${l.size} and not tracked by LFS. GitHub rejects files over 100 MB; consider LFS.`,
        });
    }
    return { rules, pointers, trackedRealFiles: trackedReal, largeUntracked, gitLfs, findings };
}
/* ------------------------------------------------------------------ */
/* 4. Missing model references                                          */
/* ------------------------------------------------------------------ */
const LIT_RE = /["'`]([^"'`\n]{1,300}?\.(?:onnx|ort|tflite|pt|pth|h5|pb|safetensors))["'`]/g;
const LABEL_RE = /["'`]([^"'`\n]{1,300}?(?:label|class|species|taxonom)[^"'`\n]*?\.(?:json|txt|csv))["'`]/gi;
const JOIN_RE = /path\.(?:join|resolve)\(((?:[^()]|\([^()]*\))*?\.(?:onnx|ort|tflite|pt|pth)(?:[^()]|\([^()]*\))*?)\)/g;
function candidates(fileRel, ref) {
    const clean = ref.replace(/^\.\//, "");
    const list = ref.startsWith("/")
        ? [ref.slice(1), "public" + ref]
        : [path.posix.join(path.posix.dirname(fileRel), ref), clean, "public/" + clean];
    if (ref.startsWith("~/") || ref.startsWith("@/"))
        list.push(ref.slice(2), "src/" + ref.slice(2));
    return list.map((p) => path.posix.normalize(p));
}
export function findMissingModelReferences() {
    const findings = [];
    const all = allFiles();
    const byBase = new Map();
    for (const f of all) {
        const b = path.posix.basename(f.rel);
        const arr = byBase.get(b);
        if (arr)
            arr.push(f.rel);
        else
            byBase.set(b, [f.rel]);
    }
    const refs = [];
    const seen = new Set();
    const add = (r) => {
        const k = `${r.file}:${r.line}:${r.ref}`;
        if (!seen.has(k)) {
            seen.add(k);
            refs.push(r);
        }
    };
    for (const f of sourceFiles()) {
        let text;
        try {
            text = fs.readFileSync(f.abs, "utf8");
        }
        catch {
            continue;
        }
        const joined = [];
        for (const m of text.matchAll(JOIN_RE)) {
            const parts = [...m[1].matchAll(/["'`]([^"'`]+)["'`]/g)].map((x) => x[1]);
            if (parts.length) {
                const line = lineOf(text, m.index ?? 0);
                const ref = parts.join("/");
                joined.push({ line, ref });
                add({ file: f.rel, line, ref, kind: "path.join" });
            }
        }
        for (const m of text.matchAll(LIT_RE)) {
            const line = lineOf(text, m.index ?? 0);
            // A literal that is only the tail of a path.join(...) on the same line is already covered above.
            if (joined.some((j) => j.line === line && (j.ref === m[1] || j.ref.endsWith("/" + m[1]))))
                continue;
            add({ file: f.rel, line, ref: m[1], kind: "literal" });
        }
        for (const m of text.matchAll(LABEL_RE)) {
            add({ file: f.rel, line: lineOf(text, m.index ?? 0), ref: m[1], kind: "label" });
        }
    }
    const results = refs.slice(0, 300).map((r) => {
        if (/^https?:\/\//i.test(r.ref))
            return { ...r, status: "remote-url" };
        if (r.ref.includes("${"))
            return { ...r, status: "dynamic" };
        for (const c of candidates(r.file, r.ref)) {
            if (c.startsWith(".."))
                continue;
            const abs = path.join(REPO_ROOT, c);
            try {
                if (fs.existsSync(abs) && fs.statSync(abs).isFile()) {
                    const ptr = parseLfsPointer(readHead(abs, 1024));
                    return { ...r, status: ptr ? "lfs-pointer" : "found", resolvedTo: c };
                }
            }
            catch {
                /* ignore */
            }
        }
        const alt = byBase.get(path.posix.basename(r.ref)) ?? [];
        return { ...r, status: "missing", similarFiles: alt.slice(0, 5) };
    });
    const missing = results.filter((r) => r.status === "missing");
    const pointerRefs = results.filter((r) => r.status === "lfs-pointer");
    for (const m of missing.slice(0, 15)) {
        findings.push({
            severity: "error",
            area: "model-references",
            message: `${m.file}:${m.line} references "${m.ref}" but no such file exists in the repo.`,
            evidence: m.similarFiles?.length ? { similarFiles: m.similarFiles } : undefined,
        });
    }
    for (const p of pointerRefs.slice(0, 10)) {
        findings.push({
            severity: "warn",
            area: "model-references",
            message: `${p.file}:${p.line} references "${p.ref}", which resolves to an LFS pointer (${p.resolvedTo}).`,
        });
    }
    if (refs.length === 0) {
        findings.push({
            severity: "warn",
            area: "model-references",
            message: "No model/label file references found in source code, so nothing could be verified.",
        });
    }
    const referencedBases = new Set(results.map((r) => path.posix.basename(r.ref)));
    for (const m of findModelFiles().models) {
        if (!referencedBases.has(path.posix.basename(m.path))) {
            findings.push({
                severity: "info",
                area: "model-references",
                message: `${m.path} exists but is not referenced by any string literal in the source.`,
            });
        }
    }
    return { totalReferences: refs.length, results, findings };
}
/* ------------------------------------------------------------------ */
/* 5. Image upload code                                                 */
/* ------------------------------------------------------------------ */
const UPLOAD_PATTERNS = [
    ["formData", /\.formData\(\)|new FormData\(/i],
    ["multipartLibs", /\b(multer|formidable|busboy|multiparty|express-fileupload|@fastify\/multipart)\b/i],
    ["fileInput", /type=\{?["']file["']/i],
    ["acceptAttr", /\baccept=/i],
    ["typeValidation", /\.(mimetype|type)\s*(===|!==|\.startsWith|\.includes)|content-type/i],
    ["sizeLimit", /bodySizeLimit|sizeLimit|maxFileSize|fileSize\s*:|\.size\s*(>|>=|<|<=)\s*[\d(]|MAX_(FILE|UPLOAD|IMAGE)/i],
    ["clientResize", /createImageBitmap|canvas|\.toBlob\(|\.toDataURL\(|browser-image-compression|compressorjs/i],
    ["serverImageProcessing", /\bsharp\b|\bjimp\b|pngjs|@napi-rs\/canvas|node-canvas/i],
    ["base64Upload", /base64|readAsDataURL/i],
    ["bodyParserConfig", /bodyParser\s*:/i],
    ["edgeRuntime", /runtime\s*=\s*["']edge["']/i],
    ["localFsWrite", /fs\.(?:promises\.)?(?:writeFile|writeFileSync|createWriteStream|mkdir)/i],
    ["inferenceCall", /InferenceSession|ort\.Tensor|new Tensor\(/i],
];
export function checkImageUpload() {
    const findings = [];
    const files = sourceFiles();
    const hits = {};
    for (const [id, re] of UPLOAD_PATTERNS)
        hits[id] = grepFiles(files, re, 5);
    const has = (id) => hits[id].length > 0;
    const serverSideUpload = has("multipartLibs") || hits["formData"].some((h) => /(^|\/)(api|route)/i.test(h.file));
    if (!has("formData") && !has("multipartLibs") && !has("fileInput") && !has("base64Upload")) {
        findings.push({
            severity: "info",
            area: "upload",
            message: "No obvious image-upload code found (FormData, multipart libs, file input, base64).",
        });
    }
    if (serverSideUpload && !has("sizeLimit")) {
        findings.push({
            severity: "warn",
            area: "upload",
            message: "Server-side upload handling found but no file-size limit or check was detected.",
        });
    }
    if (serverSideUpload && !has("clientResize")) {
        findings.push({
            severity: "warn",
            area: "upload",
            message: "Images are posted to a server route with no client-side resize/compress step. Vercel Functions have a request body " +
                "limit of about 4.5 MB (check current Vercel docs); phone photos often exceed it. Resize in the browser before upload.",
        });
    }
    if (has("fileInput") && !has("acceptAttr")) {
        findings.push({ severity: "info", area: "upload", message: "File input found without an accept attribute." });
    }
    if ((has("fileInput") || serverSideUpload) && !has("typeValidation")) {
        findings.push({ severity: "warn", area: "upload", message: "No MIME/type validation of uploaded files was detected." });
    }
    if (has("localFsWrite")) {
        findings.push({
            severity: "warn",
            area: "upload",
            message: "Code writes to the local filesystem. On Vercel the filesystem is read-only except /tmp, and /tmp is not shared across invocations.",
            evidence: hits["localFsWrite"],
        });
    }
    if (has("edgeRuntime") && (has("serverImageProcessing") || has("inferenceCall"))) {
        findings.push({
            severity: "error",
            area: "upload",
            message: "A route is set to the edge runtime but image processing / ONNX inference is used. Native modules (sharp, onnxruntime-node) do not run on the edge runtime.",
            evidence: hits["edgeRuntime"],
        });
    }
    if (has("base64Upload")) {
        findings.push({
            severity: "info",
            area: "upload",
            message: "Base64 encoding adds roughly 33% to payload size, which counts toward request size limits.",
        });
    }
    // Compare the model's expected input size with preprocessing code.
    try {
        const m = findModelFiles().models.find((x) => /\.onnx$/i.test(x.path) && !x.lfsPointer);
        if (m) {
            const info = inspectOnnx(m.path);
            const first = "summary" in info && info.summary ? info.summary.inputs[0] : undefined;
            const dims = (first?.shape ?? []).filter((d) => typeof d === "number" && d > 3);
            for (const d of new Set(dims)) {
                if (grepFiles(files, new RegExp(`\\b${d}\\b`), 1).length === 0) {
                    findings.push({
                        severity: "warn",
                        area: "upload",
                        message: `Model input "${first?.name}" expects a dimension of ${d}, but the number ${d} does not appear in any source file. Preprocessing may resize to the wrong size.`,
                        evidence: { modelInputShape: first?.shape },
                    });
                }
            }
        }
    }
    catch {
        /* model not inspectable; other tools report that */
    }
    return { hits, findings };
}
/* ------------------------------------------------------------------ */
/* 6. Vercel deployment                                                 */
/* ------------------------------------------------------------------ */
function ignoredBy(content, rel) {
    if (!content)
        return null;
    for (const raw of content.split(/\r?\n/)) {
        const s = raw.trim();
        if (!s || s.startsWith("#") || s.startsWith("!"))
            continue;
        const re = globToRegExp(s.replace(/\/+$/, ""));
        const segs = rel.split("/");
        for (let i = 0; i < segs.length; i++) {
            if (re.test(segs.slice(0, i + 1).join("/")))
                return s;
        }
    }
    return null;
}
export function analyzeVercel() {
    const findings = [];
    const readJson = (rel) => {
        const t = readText(rel);
        if (t === null)
            return null;
        try {
            return JSON.parse(t);
        }
        catch (e) {
            findings.push({ severity: "error", area: "vercel", message: `${rel} is not valid JSON: ${e.message}` });
            return null;
        }
    };
    const vercelJson = readJson("vercel.json");
    const pkg = readJson("package.json");
    const deps = { ...(pkg?.dependencies ?? {}), ...(pkg?.devDependencies ?? {}) };
    const nextCfgFile = ["next.config.js", "next.config.mjs", "next.config.cjs", "next.config.ts"].find((f) => fs.existsSync(path.join(REPO_ROOT, f)));
    const nextCfg = nextCfgFile ? readText(nextCfgFile) ?? "" : "";
    const requirements = readText("requirements.txt");
    const vercelignore = readText(".vercelignore");
    const gitignore = readText(".gitignore");
    const gitattributes = readText(".gitattributes");
    const usesOrtNode = "onnxruntime-node" in deps;
    const usesOrtWeb = "onnxruntime-web" in deps;
    const usesNext = "next" in deps;
    const pythonOrt = /onnxruntime/i.test(requirements ?? "");
    const serverSideInference = usesOrtNode || pythonOrt;
    const models = findModelFiles().models;
    const totalModelBytes = models.reduce((s, m) => s + (m.lfsPointer ? 0 : m.sizeBytes), 0);
    if (!vercelJson) {
        findings.push({ severity: "info", area: "vercel", message: "No vercel.json found; Vercel defaults apply." });
    }
    const fnConfig = JSON.stringify(vercelJson?.functions ?? {});
    const hasMaxDuration = /maxDuration/.test(fnConfig) || /maxDuration/.test(nextCfg) || /maxDuration/.test(allSourceText());
    const tracesModels = /includeFiles/.test(fnConfig) || (/outputFileTracingIncludes/.test(nextCfg) && /onnx|models?/i.test(nextCfg));
    if (serverSideInference) {
        if (!hasMaxDuration) {
            findings.push({
                severity: "warn",
                area: "vercel",
                message: "Server-side ONNX inference with no maxDuration configured. Cold start + model load can exceed the default function timeout. " +
                    "Set functions.<glob>.maxDuration in vercel.json (or 'export const maxDuration' in the route).",
            });
        }
        const serverModels = models.filter((m) => !m.path.startsWith("public/"));
        if (serverModels.length && !tracesModels) {
            findings.push({
                severity: "warn",
                area: "vercel",
                message: "Models outside public/ are read at runtime but no file-tracing include was found. Files read via fs are only deployed if traced: " +
                    "use functions.<glob>.includeFiles in vercel.json, or outputFileTracingIncludes in next.config " +
                    "(under 'experimental' on Next 14, top-level on Next 15+; check your version).",
                evidence: serverModels.map((m) => m.path),
            });
        }
    }
    if (usesOrtNode && usesNext) {
        if (!(/serverExternalPackages|serverComponentsExternalPackages/.test(nextCfg) && /onnxruntime-node/.test(nextCfg))) {
            findings.push({
                severity: "warn",
                area: "vercel",
                message: "onnxruntime-node is used with Next.js but is not listed in serverExternalPackages " +
                    "(experimental.serverComponentsExternalPackages on Next 13/14). Bundling its native .node binary commonly breaks the build or runtime.",
            });
        }
    }
    if (usesOrtNode) {
        findings.push({
            severity: "info",
            area: "vercel",
            message: "onnxruntime-node ships native binaries for several platforms and can push a function toward Vercel's size limit (about 250 MB unzipped). " +
                "Consider onnxruntime-web (WASM), or excluding other platforms' binaries via outputFileTracingExcludes.",
        });
    }
    if (usesOrtWeb) {
        findings.push({
            severity: "info",
            area: "vercel",
            message: "onnxruntime-web needs its .wasm files served at runtime. Check ort.env.wasm.wasmPaths and that the files are copied to public/ or a CDN path.",
        });
    }
    if (pythonOrt) {
        findings.push({
            severity: "info",
            area: "vercel",
            message: "Python + onnxruntime detected. Keep the deployed bundle under Vercel's function size limit (about 250 MB unzipped).",
        });
    }
    if (totalModelBytes > 200 * 1024 * 1024 && serverSideInference) {
        findings.push({
            severity: "warn",
            area: "vercel",
            message: `Model files total ${fmtBytes(totalModelBytes)}; bundling them in a function risks exceeding Vercel's size limit.`,
        });
    }
    for (const m of models) {
        const byVercelignore = ignoredBy(vercelignore, m.path);
        const byGitignore = ignoredBy(gitignore, m.path);
        if (byVercelignore) {
            findings.push({
                severity: "error",
                area: "vercel",
                message: `${m.path} appears to be excluded by .vercelignore (rule: ${byVercelignore}), so it is not deployed.`,
            });
        }
        if (byGitignore) {
            findings.push({
                severity: "error",
                area: "vercel",
                message: `${m.path} appears to be matched by .gitignore (rule: ${byGitignore}); it will not exist in a Git-based deployment.`,
            });
        }
    }
    if (gitattributes && /filter=lfs/.test(gitattributes)) {
        findings.push({
            severity: "warn",
            area: "vercel",
            message: "Repo uses Git LFS. Verify Git LFS is enabled in the Vercel project settings (Settings, Git section); otherwise builds check out pointer files instead of real models. This cannot be verified from the repo.",
        });
    }
    if (pkg && !pkg?.scripts?.build) {
        findings.push({ severity: "info", area: "vercel", message: "package.json has no build script." });
    }
    if (pkg && !pkg?.engines?.node) {
        findings.push({
            severity: "info",
            area: "vercel",
            message: "package.json has no engines.node; Vercel picks its default Node version, which can differ from your local one (relevant for native modules).",
        });
    }
    // Environment variables referenced in code
    const envNames = new Set();
    for (const f of sourceFiles()) {
        let text;
        try {
            text = fs.readFileSync(f.abs, "utf8");
        }
        catch {
            continue;
        }
        for (const m of text.matchAll(/process\.env\.([A-Z0-9_]+)/g))
            envNames.add(m[1]);
        for (const m of text.matchAll(/(?:os\.environ(?:\.get)?|os\.getenv)\s*[([]\s*["']([A-Z0-9_]+)["']/g))
            envNames.add(m[1]);
    }
    const envList = [...envNames].filter((n) => !["NODE_ENV", "VERCEL", "VERCEL_ENV", "VERCEL_URL"].includes(n)).sort();
    if (envList.length) {
        findings.push({
            severity: "info",
            area: "vercel",
            message: "Environment variables referenced in code must be set in the Vercel project (Production and Preview).",
            evidence: envList,
        });
    }
    const cap = (s) => (s === undefined ? undefined : JSON.parse(JSON.stringify(s)));
    return {
        detected: {
            framework: usesNext ? "next" : Object.keys(deps).find((d) => ["vite", "react-scripts", "nuxt", "@sveltejs/kit"].includes(d)) ?? "unknown",
            nextVersion: deps["next"],
            onnxruntimeNode: deps["onnxruntime-node"],
            onnxruntimeWeb: deps["onnxruntime-web"],
            pythonOnnxruntime: pythonOrt,
            nextConfigFile: nextCfgFile ?? null,
            nodeEngine: pkg?.engines?.node ?? null,
            hasVercelIgnore: vercelignore !== null,
        },
        vercelJson: cap(vercelJson),
        nextConfigMentions: {
            outputFileTracingIncludes: /outputFileTracingIncludes/.test(nextCfg),
            outputFileTracingExcludes: /outputFileTracingExcludes/.test(nextCfg),
            serverExternalPackages: /serverExternalPackages|serverComponentsExternalPackages/.test(nextCfg),
            outputStandalone: /output\s*:\s*["']standalone["']/.test(nextCfg),
        },
        findings,
    };
}
function allSourceText() {
    // Only used to detect "maxDuration" exports in route files.
    let out = "";
    for (const f of sourceFiles()) {
        if (!/(^|\/)(api|app)\//.test(f.rel))
            continue;
        try {
            out += fs.readFileSync(f.abs, "utf8").slice(0, 20_000) + "\n";
        }
        catch {
            /* ignore */
        }
    }
    return out;
}
/* ------------------------------------------------------------------ */
/* 7. Combined diagnosis                                                */
/* ------------------------------------------------------------------ */
const SEV_ORDER = { error: 0, warn: 1, info: 2 };
export async function diagnoseModelLoading() {
    const findings = [];
    const { models, labelFiles } = findModelFiles();
    if (models.length === 0) {
        findings.push({
            severity: "error",
            area: "model-files",
            message: "No model files (.onnx, .tflite, .pt, ...) found in the repo.",
        });
    }
    if (labelFiles.length === 0) {
        findings.push({
            severity: "info",
            area: "model-files",
            message: "No label/class-name file found (looked for names containing label, class, species, taxonomy).",
        });
    }
    for (const m of models.filter((x) => /\.onnx$/i.test(x.path) && !x.lfsPointer)) {
        const r = inspectOnnx(m.path);
        if (!r.valid) {
            findings.push({ severity: "error", area: "onnx", message: `${m.path}: ${r.problem}`, evidence: r });
            continue;
        }
        const s = r.summary;
        if (!s)
            continue;
        findings.push({
            severity: "info",
            area: "onnx",
            message: `${m.path}: IR v${s.irVersion}, opset ${s.opsets.map((o) => `${o.domain}:${o.version}`).join(", ")}, ${s.nodeCount} nodes.`,
            evidence: { inputs: s.inputs, outputs: s.outputs },
        });
        for (const ext of s.externalData) {
            const sidecar = path.join(path.dirname(resolveInRepo(m.path)), ext.location);
            if (!fs.existsSync(sidecar)) {
                findings.push({
                    severity: "error",
                    area: "onnx",
                    message: `${m.path} stores weights in external file "${ext.location}" (tensor ${ext.name}) but that file is missing next to the model.`,
                });
            }
        }
    }
    findings.push(...(await inspectLfs()).findings);
    findings.push(...findMissingModelReferences().findings);
    findings.push(...analyzeVercel().findings.filter((f) => f.severity !== "info"));
    findings.push(...checkImageUpload().findings.filter((f) => f.severity !== "info"));
    findings.sort((a, b) => SEV_ORDER[a.severity] - SEV_ORDER[b.severity]);
    const areas = new Set(findings.filter((f) => f.severity === "error").map((f) => f.area));
    const nextSteps = [];
    if (areas.has("lfs")) {
        nextSteps.push("Run 'git lfs ls-files' and 'git lfs pull'. Enable Git LFS in the Vercel project settings, then redeploy (clear the build cache).");
    }
    if (areas.has("model-references"))
        nextSteps.push("Fix or add the missing files listed under model-references (see similarFiles hints).");
    if (areas.has("onnx"))
        nextSteps.push("Re-export or re-download the model, then re-run inspect_onnx_model to confirm it parses.");
    if (areas.has("vercel"))
        nextSteps.push("Apply the vercel findings: file tracing includes, external packages, ignore rules.");
    if (areas.has("upload"))
        nextSteps.push("Fix the upload findings, especially runtime or filesystem problems.");
    if (nextSteps.length === 0) {
        nextSteps.push("No blocking problems found statically. Paste the exact runtime error into explain_model_error, and compare local vs. Vercel build logs.");
    }
    return {
        counts: {
            error: findings.filter((f) => f.severity === "error").length,
            warn: findings.filter((f) => f.severity === "warn").length,
            info: findings.filter((f) => f.severity === "info").length,
        },
        modelFiles: models,
        findings,
        suggestedNextSteps: nextSteps,
    };
}
/* ------------------------------------------------------------------ */
/* 8. Error explainer                                                   */
/* ------------------------------------------------------------------ */
const ERROR_TABLE = [
    {
        match: /protobuf parsing failed|invalid wire type|failed to parse|ModelProto/i,
        title: "Model file is not a valid ONNX protobuf",
        causes: [
            "File is a Git LFS pointer (about 130 bytes of text) because LFS content was not pulled during checkout or deploy.",
            "File is truncated or corrupted (interrupted download or upload).",
            "File is an HTML error page saved with a .onnx extension, or a different format.",
        ],
        tools: ["inspect_onnx_model", "inspect_git_lfs"],
    },
    {
        match: /ENOENT|no such file|NO_SUCHFILE|cannot find (the )?file|failed to load model.*(not exist|not found)/i,
        title: "Model path does not exist at runtime",
        causes: [
            "Path is wrong or relative to a different working directory (on Vercel, cwd is the function root).",
            "File is not included in the serverless bundle (needs includeFiles or outputFileTracingIncludes).",
            "File is excluded by .gitignore or .vercelignore.",
        ],
        tools: ["find_missing_model_references", "analyze_vercel_config"],
    },
    {
        match: /Cannot find module ['"]?onnxruntime|MODULE_NOT_FOUND/i,
        title: "onnxruntime package is missing at runtime",
        causes: [
            "Package is only in devDependencies or was not installed in the deploy.",
            "Package was bundled instead of kept external (serverExternalPackages).",
            "Wrong package for the environment (onnxruntime-node vs onnxruntime-web).",
        ],
        tools: ["analyze_vercel_config"],
    },
    {
        match: /GLIBC|invalid ELF header|ERR_DLOPEN_FAILED|onnxruntime_binding\.node|not a valid Win32/i,
        title: "Native binary does not match the runtime platform",
        causes: [
            "node_modules built on macOS/Windows were deployed to Linux (do not commit node_modules).",
            "Node version differs between build and runtime; pin engines.node.",
            "Architecture mismatch (arm64 vs x64).",
        ],
        tools: ["analyze_vercel_config"],
    },
    {
        match: /unsupported model IR version|opset|No Op registered|NOT_IMPLEMENTED|Unrecognized attribute/i,
        title: "Model is newer than the onnxruntime build",
        causes: [
            "Model IR version or opset is higher than the installed onnxruntime supports.",
            "Model uses custom or contrib operators not compiled into this runtime.",
        ],
        tools: ["inspect_onnx_model"],
    },
    {
        match: /out of memory|heap|OOM|bad_alloc|allocation failed|signal: killed|exceeded.*memory/i,
        title: "Out of memory while loading or running the model",
        causes: [
            "Function memory is too small for the model plus runtime.",
            "Session is created on every request instead of being cached at module scope.",
        ],
        tools: ["inspect_onnx_model", "analyze_vercel_config"],
    },
    {
        match: /FUNCTION_INVOCATION_TIMEOUT|task timed out|504/i,
        title: "Function timed out",
        causes: ["Cold start plus model load exceeds the timeout; raise maxDuration and reuse the inference session."],
        tools: ["analyze_vercel_config"],
    },
    {
        match: /FUNCTION_PAYLOAD_TOO_LARGE|413|request entity too large|body exceeded/i,
        title: "Upload is too large for the function",
        causes: ["Vercel Functions cap the request body (about 4.5 MB); resize images in the browser before upload."],
        tools: ["check_image_upload"],
    },
    {
        match: /invalid input|unexpected input|input.*name|invalid dimensions|invalid rank|shape mismatch|got invalid/i,
        title: "Input name or shape does not match the model",
        causes: [
            "Feed key differs from the model's input name.",
            "Tensor dims differ from the model's expected shape (NCHW vs NHWC, wrong resize size).",
        ],
        tools: ["inspect_onnx_model", "check_image_upload"],
    },
    {
        match: /unsupported data type|float16|type mismatch|expected.*(float|int64)/i,
        title: "Tensor dtype mismatch",
        causes: ["Feeding the wrong typed array (e.g. Float32Array where the model wants int64 as BigInt64Array, or float16)."],
        tools: ["inspect_onnx_model"],
    },
    {
        match: /Unexpected token '<'|<!DOCTYPE/i,
        title: "Got an HTML page where a model or JSON was expected",
        causes: ["The model/label URL returns a 404 or login page; check the URL and that the file is deployed."],
        tools: ["find_missing_model_references", "analyze_vercel_config"],
    },
    {
        match: /failed to fetch|fetch failed|CORS|wasm/i,
        title: "Browser could not fetch the model or WASM files",
        causes: [
            "Wrong URL or path under public/.",
            "onnxruntime-web WASM files not served (ort.env.wasm.wasmPaths).",
            "CORS or content-type issues on a remote model URL.",
        ],
        tools: ["find_missing_model_references", "analyze_vercel_config"],
    },
    {
        match: /exceeds the maximum size|unzipped size|250\s?MB/i,
        title: "Serverless function bundle too large",
        causes: [
            "Model plus onnxruntime-node binaries exceed the size limit; serve the model from storage or a CDN, use onnxruntime-web, or exclude unused platform binaries.",
        ],
        tools: ["find_model_files", "analyze_vercel_config"],
    },
];
export function explainModelError(errorText) {
    const matches = ERROR_TABLE.filter((e) => e.match.test(errorText)).map((e) => ({
        diagnosis: e.title,
        likelyCauses: e.causes,
        runTheseTools: e.tools,
    }));
    return matches.length
        ? { matches }
        : {
            matches: [],
            note: "No known pattern matched. Run debug_model_loading for a full static check and compare local vs. deployed behavior.",
        };
}
