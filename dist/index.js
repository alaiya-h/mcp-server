#!/usr/bin/env node
import fs from "node:fs";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { REPO_ROOT, allFiles, walk, resolveInRepo, readTextRange, grepFiles, globToRegExp, escapeRegExp, fmtBytes, } from "./util.js";
import { findModelFiles, inspectOnnx, inspectLfs, findMissingModelReferences, checkImageUpload, analyzeVercel, diagnoseModelLoading, explainModelError, } from "./analyzers.js";
const MAX_OUTPUT_CHARS = 80_000;
/** Wrap a tool body: JSON-serialise results, convert exceptions into MCP tool errors. */
async function run(fn) {
    try {
        const data = await fn();
        let text = typeof data === "string" ? data : JSON.stringify(data, null, 2);
        if (text.length > MAX_OUTPUT_CHARS) {
            text = text.slice(0, MAX_OUTPUT_CHARS) + "\n...[output truncated]";
        }
        return { content: [{ type: "text", text }] };
    }
    catch (e) {
        return {
            content: [{ type: "text", text: `Error: ${e instanceof Error ? e.message : String(e)}` }],
            isError: true,
        };
    }
}
const READ_ONLY = { readOnlyHint: true, openWorldHint: false };
const server = new McpServer({ name: "herbify-debug", version: "1.0.0" });
/* ---------------------------- file access ---------------------------- */
server.registerTool("list_files", {
    title: "List files",
    description: "List files and folders in the Herbify repo (skips node_modules, .git, .next, dist, etc.). Paths are relative to the repo root.",
    inputSchema: {
        path: z.string().optional().describe("Directory relative to repo root. Default: repo root."),
        maxDepth: z.number().int().min(1).max(10).optional().describe("Directory depth. Default 2."),
    },
    annotations: READ_ONLY,
}, async ({ path: p, maxDepth }) => run(() => {
    const start = resolveInRepo(p ?? ".");
    if (!fs.existsSync(start) || !fs.statSync(start).isDirectory()) {
        throw new Error(`Not a directory: ${p}`);
    }
    const entries = [...walk(start, maxDepth ?? 2)];
    const lines = entries
        .slice(0, 500)
        .map((e) => (e.isDir ? `${e.rel}/` : `${e.rel}  (${fmtBytes(e.size)})`));
    if (entries.length > 500)
        lines.push(`... ${entries.length - 500} more entries`);
    return `Repo root: ${REPO_ROOT}\n${lines.join("\n")}`;
}));
server.registerTool("read_file", {
    title: "Read file",
    description: "Read a text file from the repo with line numbers. Refuses binary files (use the model tools for those).",
    inputSchema: {
        path: z.string().describe("File path relative to repo root."),
        startLine: z.number().int().min(1).optional(),
        endLine: z.number().int().min(1).optional(),
    },
    annotations: READ_ONLY,
}, async ({ path: p, startLine, endLine }) => run(() => {
    const abs = resolveInRepo(p);
    if (!fs.existsSync(abs))
        throw new Error(`File not found: ${p}`);
    return readTextRange(abs, startLine ?? 1, endLine);
}));
server.registerTool("search_files", {
    title: "Search files",
    description: "Search text files in the repo for a string or regular expression. Returns file, line and matching text.",
    inputSchema: {
        query: z.string().min(1).describe("Text or regex to look for."),
        isRegex: z.boolean().optional().describe("Treat query as a regular expression. Default false."),
        caseSensitive: z.boolean().optional().describe("Default false."),
        pathGlob: z.string().optional().describe("Only search paths matching this glob, e.g. 'src/**/*.ts' or '*.json'."),
        maxResults: z.number().int().min(1).max(200).optional().describe("Default 50."),
    },
    annotations: READ_ONLY,
}, async ({ query, isRegex, caseSensitive, pathGlob, maxResults }) => run(() => {
    const re = new RegExp(isRegex ? query : escapeRegExp(query), caseSensitive ? "" : "i");
    const glob = pathGlob ? globToRegExp(pathGlob) : null;
    const files = allFiles().filter((e) => !glob || glob.test(e.rel));
    const hits = grepFiles(files, re, maxResults ?? 50);
    return { query, matches: hits.length, hits };
}));
/* ------------------------------ model ------------------------------- */
server.registerTool("find_model_files", {
    title: "Find model files",
    description: "Locate model files (.onnx, .tflite, .pt, ...) and label files, with sizes and whether each is a Git LFS pointer instead of the real file.",
    inputSchema: {},
    annotations: READ_ONLY,
}, async () => run(() => findModelFiles()));
server.registerTool("inspect_onnx_model", {
    title: "Inspect ONNX model",
    description: "Parse an ONNX file header: IR version, opset, producer, input/output names, dtypes and shapes, operator counts, external-data files. " +
        "Detects LFS pointers, HTML-instead-of-model and corrupted files. Defaults to the first .onnx found.",
    inputSchema: {
        path: z.string().optional().describe("Model path relative to repo root. Default: first .onnx in the repo."),
    },
    annotations: READ_ONLY,
}, async ({ path: p }) => run(() => inspectOnnx(p)));
server.registerTool("find_missing_model_references", {
    title: "Find missing model references",
    description: "Scan source code for string literals and path.join calls that point at model or label files, and check each exists in the repo " +
        "(also flags LFS pointers, remote URLs and dynamic paths).",
    inputSchema: {},
    annotations: READ_ONLY,
}, async () => run(() => findMissingModelReferences()));
server.registerTool("explain_model_error", {
    title: "Explain model loading error",
    description: "Paste an error message from onnxruntime, Node or Vercel logs. Returns likely causes and which tools of this server to run next.",
    inputSchema: {
        errorText: z.string().min(1).describe("The error message or log excerpt."),
    },
    annotations: READ_ONLY,
}, async ({ errorText }) => run(() => explainModelError(errorText)));
server.registerTool("debug_model_loading", {
    title: "Debug model loading (full check)",
    description: "Run all static checks (model files, ONNX validity, Git LFS, missing references, Vercel config, upload code) and return " +
        "prioritised findings with suggested next steps. Start here.",
    inputSchema: {},
    annotations: READ_ONLY,
}, async () => run(() => diagnoseModelLoading()));
/* ---------------------- upload / deployment / LFS -------------------- */
server.registerTool("check_image_upload", {
    title: "Check image upload code",
    description: "Find image-upload handling (FormData, multer/formidable, file inputs, base64) and check for missing size limits, type validation, " +
        "client-side resizing, filesystem writes, edge-runtime conflicts, and preprocessing size vs. the model's input shape.",
    inputSchema: {},
    annotations: READ_ONLY,
}, async () => run(() => checkImageUpload()));
server.registerTool("analyze_vercel_config", {
    title: "Analyze Vercel deployment",
    description: "Inspect vercel.json, next.config.*, package.json, requirements.txt, .vercelignore and .gitignore for problems that break " +
        "ONNX/model loading on Vercel (file tracing, external packages, timeouts, size limits, ignored models, env vars).",
    inputSchema: {},
    annotations: READ_ONLY,
}, async () => run(() => analyzeVercel()));
server.registerTool("inspect_git_lfs", {
    title: "Inspect Git LFS",
    description: "List .gitattributes LFS rules, find files that are still LFS pointers, files properly checked out, large untracked files, " +
        "and (if git-lfs is installed) the output of 'git lfs ls-files'.",
    inputSchema: {},
    annotations: READ_ONLY,
}, async () => run(() => inspectLfs()));
/* -------------------------------- start ------------------------------ */
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    // stdout is reserved for the MCP protocol; log to stderr only.
    console.error(`herbify-debug MCP server running (repo: ${REPO_ROOT})`);
}
main().catch((err) => {
    console.error("Fatal:", err);
    process.exit(1);
});
