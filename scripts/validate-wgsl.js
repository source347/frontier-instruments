#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const sourcePath = path.join(__dirname, "..", "instruments", "webgpu", "webgpu.js");
const source = fs.readFileSync(sourcePath, "utf8");

function extractShader(name) {
  const match = source.match(new RegExp("const " + name + "=\\x60([\\s\\S]*?)\\x60;"));
  if (!match) throw new Error("Could not extract " + name + " from " + sourcePath);
  return match[1].trim() + "\n";
}

const tempDir = fs.mkdtempSync(path.join(process.cwd(), ".wgsl-validation-"));
const shaders = [
  ["vector-add", extractShader("shaderAdd")],
  ["parallel-reduction", extractShader("shaderReduce")]
];
const workgroupSizes = [64, 128, 256];
let failures = 0;

try {
  for (const [name, code] of shaders) {
    const file = path.join(tempDir, name + ".wgsl");
    fs.writeFileSync(file, code);
    for (const size of workgroupSizes) {
      console.log("\nValidating " + name + " (WG_SIZE=" + size + ")");
      const result = spawnSync("npx", [
        "--yes",
        "naga-wasi-cli@0.1.0",
        "--override",
        "WG_SIZE=" + size,
        path.relative(process.cwd(), file)
      ], { stdio: "inherit" });
      if (result.error) {
        console.error(result.error.message);
        failures++;
      } else if (result.status !== 0) {
        console.error("Naga rejected " + name + " with WG_SIZE=" + size);
        failures++;
      }
    }
  }
} finally {
  fs.rmSync(tempDir, { recursive: true, force: true });
}

if (failures) {
  console.error("\nWGSL validation failed in " + failures + " configuration(s).");
  process.exit(1);
}
console.log("\nAll WGSL shaders passed Naga validation for workgroup sizes 64, 128, and 256.");
