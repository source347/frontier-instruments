const { test, expect } = require("@playwright/test");
const pagePath = "/instruments/webgpu/";

async function installGpuMock(page) {
  await page.addInitScript(() => {
    window.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8 };
    window.GPUMapMode = { READ: 1 };
    window.__gpuMock = { shaders: [], dispatches: [] };
    class MockBuffer {
      constructor(size) { this.data = new ArrayBuffer(size); }
      destroy() {}
      async mapAsync() {}
      getMappedRange() { return this.data; }
      unmap() {}
    }
    const device = {
      limits: { maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: 1 << 27, maxComputeWorkgroupSizeX: 256 },
      queue: {
        writeBuffer(buffer, offset, source) {
          const bytes = source instanceof ArrayBuffer ? new Uint8Array(source) : new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
          new Uint8Array(buffer.data, offset, bytes.byteLength).set(bytes);
        },
        submit(commands) { commands.forEach(command => command.execute()); },
        async onSubmittedWorkDone() {}
      },
      createBuffer({ size }) { return new MockBuffer(size); },
      createShaderModule({ code }) { window.__gpuMock.shaders.push(code); return { code }; },
      createComputePipeline({ compute }) {
        const code = compute.module.code;
        return {
          operation: code.includes("var<workgroup> scratch") ? "reduce" : "add",
          workgroupSize: compute.constants.WG_SIZE,
          getBindGroupLayout() { return {}; }
        };
      },
      createBindGroup({ entries }) { return { entries }; },
      createCommandEncoder() {
        let dispatch = null, copy = null;
        return {
          beginComputePass() {
            return {
              setPipeline(pipeline) { dispatch = { pipeline }; },
              setBindGroup(_index, bindGroup) { dispatch.bindGroup = bindGroup; },
              dispatchWorkgroups(groups) { dispatch.groups = groups; },
              end() {}
            };
          },
          copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) { copy = { source, sourceOffset, target, targetOffset, size }; },
          finish() {
            return { execute() {
              const { pipeline, bindGroup, groups } = dispatch;
              const buffers = Object.fromEntries(bindGroup.entries.map(e => [e.binding, e.resource.buffer]));
              const out = new Float32Array(buffers[pipeline.operation === "add" ? 2 : 1].data);
              const input = new Float32Array(buffers[0].data);
              if (pipeline.operation === "add") {
                const b = new Float32Array(buffers[1].data);
                for (let i = 0; i < out.length; i++) out[i] = input[i] + b[i];
              } else {
                for (let group = 0; group < groups; group++) {
                  let sum = 0;
                  const start = group * pipeline.workgroupSize, end = Math.min(start + pipeline.workgroupSize, input.length);
                  for (let i = start; i < end; i++) sum += input[i];
                  out[group] = sum;
                }
              }
              window.__gpuMock.dispatches.push({ operation: pipeline.operation, groups, workgroupSize: pipeline.workgroupSize });
              new Uint8Array(copy.target.data, copy.targetOffset, copy.size).set(new Uint8Array(copy.source.data, copy.sourceOffset, copy.size));
            }};
          }
        };
      }
    };
    const adapter = { limits: device.limits, async requestDevice() { return device; } };
    Object.defineProperty(navigator, "gpu", { configurable: true, value: { async requestAdapter() { return adapter; } } });
  });
}

test("CPU baseline remains usable when WebGPU is unavailable", async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, "gpu", { configurable: true, value: undefined }));
  await page.goto(pagePath);
  await expect(page.locator("#device-tag")).toHaveText("UNAVAILABLE");
  await page.locator("#run-cpu").click();
  await expect(page.locator("#cpu-time")).not.toHaveText("—");
  await expect(page.locator("#output-preview")).toContainText("[0]");
  await expect(page.locator("#run-gpu")).toBeDisabled();
});

test("vector addition dispatches and validates GPU output", async ({ page }) => {
  await installGpuMock(page);
  await page.goto(pagePath);
  await expect(page.locator("#device-tag")).toHaveText("READY");
  await page.locator("#elements").selectOption("16384");
  await page.locator("#workgroup").selectOption("128");
  await page.locator("#run-gpu").click();
  await expect(page.locator("#validation")).toHaveText("PASS");
  await expect(page.locator("#output-preview")).toContainText("[0]");
  expect(await page.evaluate(() => window.__gpuMock.dispatches[0])).toEqual({ operation: "add", groups: 128, workgroupSize: 128 });
  const shader = await page.evaluate(() => window.__gpuMock.shaders[0]);
  expect(shader).toContain("@workgroup_size(WG_SIZE)");
  expect(shader).toContain("arrayLength(&output.values)");
});

test("parallel reduction uses workgroup barriers and validates its result", async ({ page }) => {
  await installGpuMock(page);
  await page.goto(pagePath);
  await expect(page.locator("#device-tag")).toHaveText("READY");
  await page.locator("#operation").selectOption("reduce");
  await page.locator("#elements").selectOption("16384");
  await page.locator("#workgroup").selectOption("64");
  await page.locator("#run-gpu").click();
  await expect(page.locator("#validation")).toHaveText("PASS");
  await expect(page.locator("#output-preview")).toContainText("Σ Aᵢ");
  expect(await page.evaluate(() => window.__gpuMock.dispatches[0])).toEqual({ operation: "reduce", groups: 256, workgroupSize: 64 });
  const shader = await page.evaluate(() => window.__gpuMock.shaders[0]);
  expect(shader).toContain("workgroupBarrier()");
  expect(shader).toContain("scratch[lid + stride]");
});
