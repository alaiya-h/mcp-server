# Herbify Debug MCP Server

A read-only [Model Context Protocol](https://modelcontextprotocol.io) server (TypeScript, official `@modelcontextprotocol/sdk`) for debugging the Herbify plant-identification repo: ONNX model loading, Git LFS, image upload code and Vercel deployment.

It inspects a **local checkout** of the repo. It never writes files, and every path is sandboxed to the repo root (including symlink escapes).

## Setup

Requirements: Node.js 18+, git, and (recommended) git-lfs.

```bash
# 1. Get the repo you want to debug
git lfs install
git clone https://github.com/alaiya-h/herbify.git
cd herbify && git lfs pull && cd ..

# 2. Build this server
cd herbify-mcp-server
npm install
npm run build
```

> **Tip:** to see what Vercel sees when LFS is not pulled, clone a second copy with
> `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/alaiya-h/herbify.git herbify-nolfs`
> and point the server at that copy. Model files will then show up as LFS pointers.

## Try it with the MCP Inspector

```bash
REPO_ROOT=/absolute/path/to/herbify npm run inspect
```

## Connect it to a client

The server uses **stdio**, so it runs locally next to your repo. Replace the paths below with absolute paths.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings, Developer, Edit Config):

```json
{
  "mcpServers": {
    "herbify-debug": {
      "command": "node",
      "args": ["/absolute/path/to/herbify-mcp-server/dist/index.js"],
      "env": { "REPO_ROOT": "/absolute/path/to/herbify" }
    }
  }
}
```

Restart Claude Desktop.

### Claude Code

```bash
claude mcp add herbify-debug \
  -e REPO_ROOT=/absolute/path/to/herbify \
  -- node /absolute/path/to/herbify-mcp-server/dist/index.js
```

### Other stdio clients (Cursor, VS Code, Windsurf, ...)

Use the same command / args / env as above in that client's MCP settings.

### ChatGPT

ChatGPT connects to **remote** MCP servers over HTTPS, not local stdio ones. This server would need an HTTP transport and a public URL with authentication. Because it exposes your source code, do not publish it unauthenticated.

## Tools

| Tool | What it does |
|---|---|
| `debug_model_loading` | Runs every check below and returns prioritised findings and next steps. **Start here.** |
| `list_files` | Directory listing (skips `node_modules`, `.git`, `.next`, `dist`, ...). |
| `read_file` | Reads a text file with line numbers and optional line range. |
| `search_files` | String or regex search, optional path glob. |
| `find_model_files` | Finds `.onnx`, `.tflite`, `.pt`, ... and label files; flags LFS pointers. |
| `inspect_onnx_model` | Parses the ONNX header: IR version, opset, inputs/outputs (name, dtype, shape), op counts, external data. Detects LFS pointers, HTML pages and corrupt files. |
| `find_missing_model_references` | Finds model/label paths in source code and checks they exist. |
| `inspect_git_lfs` | `.gitattributes` rules, remaining pointer files, `git lfs ls-files`. |
| `check_image_upload` | Reviews upload code: size limits, type validation, client resize, `/tmp` vs. filesystem writes, edge runtime, input size vs. model shape. |
| `analyze_vercel_config` | Reviews `vercel.json`, `next.config.*`, `package.json`, ignore files, file tracing, external packages, timeouts, size limits, env vars. |
| `explain_model_error` | Paste an error message and get likely causes and which tools to run next. |

## Example prompts

- "Run debug_model_loading and summarise what is wrong."
- "Inspect the ONNX model. What input shape does it expect, and does my preprocessing match?"
- "I get 'Protobuf parsing failed' on Vercel. Check Git LFS and Vercel config."
- "Find every place the code loads the model and verify the paths exist."

## How it detects the classic failure

If the real model is stored with Git LFS but the deploy checks out only pointer files, `model.onnx` becomes a ~130 byte text file starting with `version https://git-lfs.github.com/spec/v1`. onnxruntime then fails with errors like `Protobuf parsing failed`. `find_model_files`, `inspect_onnx_model` and `inspect_git_lfs` all detect this directly.

## Notes and limits

- Static analysis only: it cannot see your Vercel dashboard settings (for example whether Git LFS is enabled for the project) or runtime logs. Where that matters, the findings say what to verify manually.
- Vercel limits quoted in findings (request body about 4.5 MB, function size about 250 MB unzipped) change over time. Check current Vercel docs.
- ONNX files up to about 1.5 GB are parsed in memory; weight tensors are skipped, not decoded.
- Logs go to stderr; stdout is reserved for the MCP protocol.

## Development

```bash
npm run dev -- /absolute/path/to/herbify   # run from TypeScript with tsx
npm run build                              # compile to dist/
```
