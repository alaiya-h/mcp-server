import fs from "node:fs";
/**
 * Minimal, dependency-free ONNX reader. It walks the protobuf wire format of ModelProto and
 * extracts the information that matters when a model fails to load: IR version, opsets,
 * input/output names + dtypes + shapes, operator histogram and external-data references.
 * Weight tensors are skipped by length, never decoded.
 */
const DTYPES = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64",
    8: "string", 9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64",
    14: "complex64", 15: "complex128", 16: "bfloat16",
};
function* fields(buf, start, end) {
    let pos = start;
    const readVarint = () => {
        let result = 0n;
        let shift = 0n;
        for (let i = 0; i < 10; i++) {
            if (pos >= end)
                throw new Error("truncated varint");
            const b = buf[pos++];
            result |= BigInt(b & 0x7f) << shift;
            if (!(b & 0x80))
                return result;
            shift += 7n;
        }
        throw new Error("varint too long");
    };
    while (pos < end) {
        const tag = Number(readVarint());
        const no = Math.floor(tag / 8);
        const wire = tag % 8;
        if (no === 0)
            throw new Error("invalid field number 0");
        if (wire === 0) {
            const value = readVarint();
            yield { no, wire, value, start: pos, end: pos };
        }
        else if (wire === 1) {
            if (pos + 8 > end)
                throw new Error("truncated 64-bit field");
            yield { no, wire, value: 0n, start: pos, end: pos + 8 };
            pos += 8;
        }
        else if (wire === 2) {
            const len = Number(readVarint());
            if (len < 0 || pos + len > end)
                throw new Error("length-delimited field overruns buffer");
            yield { no, wire, value: 0n, start: pos, end: pos + len };
            pos += len;
        }
        else if (wire === 5) {
            if (pos + 4 > end)
                throw new Error("truncated 32-bit field");
            yield { no, wire, value: 0n, start: pos, end: pos + 4 };
            pos += 4;
        }
        else {
            throw new Error(`unsupported wire type ${wire}`);
        }
    }
}
function parseValueInfo(buf, start, end) {
    let name = "";
    let dtype = "unknown";
    let shape = null;
    for (const f of fields(buf, start, end)) {
        if (f.no === 1 && f.wire === 2) {
            name = buf.toString("utf8", f.start, f.end);
        }
        else if (f.no === 2 && f.wire === 2) {
            // TypeProto
            for (const t of fields(buf, f.start, f.end)) {
                if (t.wire !== 2)
                    continue;
                if (t.no === 1) {
                    // tensor_type
                    dtype = "tensor";
                    for (const tt of fields(buf, t.start, t.end)) {
                        if (tt.no === 1 && tt.wire === 0) {
                            dtype = DTYPES[Number(tt.value)] ?? `dtype#${tt.value}`;
                        }
                        else if (tt.no === 2 && tt.wire === 2) {
                            const dims = [];
                            for (const d of fields(buf, tt.start, tt.end)) {
                                if (d.no !== 1 || d.wire !== 2)
                                    continue;
                                let dim = "?";
                                for (const x of fields(buf, d.start, d.end)) {
                                    if (x.no === 1 && x.wire === 0)
                                        dim = Number(BigInt.asIntN(64, x.value));
                                    else if (x.no === 2 && x.wire === 2)
                                        dim = buf.toString("utf8", x.start, x.end);
                                }
                                dims.push(dim);
                            }
                            shape = dims;
                        }
                    }
                }
                else {
                    dtype = "non-tensor (sequence/map/optional)";
                }
            }
        }
    }
    return { name, dtype, shape };
}
export function parseOnnx(abs) {
    const sizeBytes = fs.statSync(abs).size;
    if (sizeBytes === 0)
        throw new Error("File is empty (0 bytes).");
    if (sizeBytes > 1_500_000_000) {
        throw new Error(`File is ${sizeBytes} bytes; too large to parse in memory.`);
    }
    const buf = fs.readFileSync(abs);
    const out = {
        sizeBytes,
        irVersion: null,
        producer: "",
        producerVersion: "",
        opsets: [],
        graphName: "",
        nodeCount: 0,
        topOps: [],
        initializerCount: 0,
        initializerRawBytes: 0,
        externalData: [],
        inputs: [],
        outputs: [],
        metadata: {},
    };
    const ops = {};
    const initNames = new Set();
    const rawInputs = [];
    for (const f of fields(buf, 0, buf.length)) {
        if (f.no === 1 && f.wire === 0) {
            out.irVersion = Number(f.value);
        }
        else if (f.no === 2 && f.wire === 2) {
            out.producer = buf.toString("utf8", f.start, f.end);
        }
        else if (f.no === 3 && f.wire === 2) {
            out.producerVersion = buf.toString("utf8", f.start, f.end);
        }
        else if (f.no === 8 && f.wire === 2) {
            let domain = "";
            let version = 0;
            for (const o of fields(buf, f.start, f.end)) {
                if (o.no === 1 && o.wire === 2)
                    domain = buf.toString("utf8", o.start, o.end);
                else if (o.no === 2 && o.wire === 0)
                    version = Number(o.value);
            }
            out.opsets.push({ domain: domain || "ai.onnx", version });
        }
        else if (f.no === 14 && f.wire === 2) {
            let k = "";
            let v = "";
            for (const e of fields(buf, f.start, f.end)) {
                if (e.no === 1 && e.wire === 2)
                    k = buf.toString("utf8", e.start, e.end);
                else if (e.no === 2 && e.wire === 2)
                    v = buf.toString("utf8", e.start, e.end);
            }
            if (k && Object.keys(out.metadata).length < 20)
                out.metadata[k] = v.slice(0, 300);
        }
        else if (f.no === 7 && f.wire === 2) {
            // GraphProto
            for (const g of fields(buf, f.start, f.end)) {
                if (g.wire !== 2)
                    continue;
                switch (g.no) {
                    case 1: {
                        out.nodeCount++;
                        for (const n of fields(buf, g.start, g.end)) {
                            if (n.no === 4 && n.wire === 2) {
                                const op = buf.toString("utf8", n.start, n.end);
                                ops[op] = (ops[op] ?? 0) + 1;
                            }
                        }
                        break;
                    }
                    case 2:
                        out.graphName = buf.toString("utf8", g.start, g.end);
                        break;
                    case 5: {
                        out.initializerCount++;
                        let name = "";
                        let external = false;
                        let location = "";
                        for (const t of fields(buf, g.start, g.end)) {
                            if (t.no === 8 && t.wire === 2)
                                name = buf.toString("utf8", t.start, t.end);
                            else if (t.no === 14 && t.wire === 0 && t.value === 1n)
                                external = true;
                            else if (t.no === 9 && t.wire === 2)
                                out.initializerRawBytes += t.end - t.start;
                            else if (t.no === 13 && t.wire === 2) {
                                let k = "";
                                let v = "";
                                for (const e of fields(buf, t.start, t.end)) {
                                    if (e.no === 1 && e.wire === 2)
                                        k = buf.toString("utf8", e.start, e.end);
                                    else if (e.no === 2 && e.wire === 2)
                                        v = buf.toString("utf8", e.start, e.end);
                                }
                                if (k === "location")
                                    location = v;
                            }
                        }
                        if (name)
                            initNames.add(name);
                        if (external && out.externalData.length < 50)
                            out.externalData.push({ name, location });
                        break;
                    }
                    case 11:
                        rawInputs.push(parseValueInfo(buf, g.start, g.end));
                        break;
                    case 12:
                        out.outputs.push(parseValueInfo(buf, g.start, g.end));
                        break;
                    default:
                        break;
                }
            }
        }
    }
    // Older IR versions list initializers as graph inputs too; hide those.
    out.inputs = rawInputs.filter((i) => !initNames.has(i.name));
    out.topOps = Object.entries(ops)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15)
        .map(([op, count]) => ({ op, count }));
    if (out.irVersion === null && out.nodeCount === 0 && out.opsets.length === 0) {
        throw new Error("Parsed as protobuf but contains no ONNX ModelProto fields.");
    }
    return out;
}
