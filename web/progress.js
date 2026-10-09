import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function setStatus(node, message) {
    const wasVisible = Boolean(node.infinisplatStatus.value);
    node.infinisplatStatus.value = message;
    if (wasVisible !== Boolean(message)) node.setSize([node.size[0], node.computeSize()[1]]);
    app.graph.setDirtyCanvas(true, true);
}

async function downloadStatus(node) {
    const sequence = node.infinisplatStatusRequest = (node.infinisplatStatusRequest || 0) + 1;
    if (node.infinisplatRunning) return;
    const download = node.widgets?.find(w => w.name === "download_model")?.value;
    if (!download) { setStatus(node, ""); return; }
    const linked = node.inputs?.find(i => i.name === "intrinsics")?.link != null;
    const camera = linked ? "manual" : node.widgets?.find(w => w.name === "camera_estimation")?.value || "manual";
    try {
        const response = await api.fetchApi(`/infinisplat/model_status?camera=${encodeURIComponent(camera)}`);
        if (!response.ok) return;
        const { missing } = await response.json();
        if (sequence !== node.infinisplatStatusRequest || node.infinisplatRunning) return;
        setStatus(node, missing.length ? `Downloads on run: ${missing.join(", ")}` : "");
    } catch { /* Server may be restarting. */ }
}

app.registerExtension({
    name: "InfiniSplat.Progress",
    nodeCreated(node) {
        if (node.comfyClass !== "InfiniSplatGenerate") return;
        node.infinisplatStatus = node.addCustomWidget({
            name: "infinisplat_status",
            type: "infinisplat_status",
            value: "",
            serialize: false,
            computeSize() { return [0, this.value ? 24 : -4]; },
            draw(ctx, node, width, y) {
                if (!this.value) return;
                ctx.save();
                ctx.font = "12px sans-serif";
                ctx.fillStyle = "#aaa";
                let text = this.value;
                while (text.length > 1 && ctx.measureText(text).width > width - 24) {
                    text = text.slice(0, -2) + "…";
                }
                ctx.fillText(text, 12, y + 16);
                ctx.restore();
            },
        });
        for (const name of ["download_model", "camera_estimation"]) {
            const widget = node.widgets?.find(w => w.name === name);
            if (!widget) continue;
            const callback = widget.callback;
            widget.callback = function (...args) { callback?.apply(this, args); downloadStatus(node); };
        }
        for (const name of ["onConfigure", "onConnectionsChange"]) {
            const callback = node[name];
            node[name] = function (...args) { callback?.apply(this, args); downloadStatus(node); };
        }
        downloadStatus(node);
    },
});

api.addEventListener("infinisplat_progress", ({ detail }) => {
    const node = app.graph?.getNodeById(detail.node_id);
    if (!node?.infinisplatStatus) return;
    if (["Complete", "Interrupted or failed", "Inference failed"].includes(detail.message)) {
        node.infinisplatRunning = false;
        setStatus(node, "");
        downloadStatus(node);
    } else {
        node.infinisplatRunning = true;
        setStatus(node, detail.message);
    }
});
