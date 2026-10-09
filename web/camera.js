import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "InfiniSplat.CameraControls",
    nodeCreated(node) {
        if (node.comfyClass !== "InfiniSplatGenerate") return;
        const mode = node.widgets?.find(w => w.name === "camera_estimation");
        const focal = node.widgets?.find(w => w.name === "focal_length_mm");
        if (!mode || !focal) return;
        const originalType = focal.type;
        const originalSize = focal.computeSize;
        const update = () => {
            const linked = node.inputs?.find(i => i.name === "intrinsics")?.link != null;
            const visible = mode.value === "manual" && !linked;
            focal.type = visible ? originalType : "infinisplat_hidden";
            focal.computeSize = visible ? originalSize : () => [0, -4];
            node.setSize([node.size[0], node.computeSize()[1]]);
            app.graph.setDirtyCanvas(true, true);
        };
        const callback = mode.callback;
        mode.callback = function (...args) { callback?.apply(this, args); update(); };
        for (const name of ["onConfigure", "onConnectionsChange"]) {
            const callback = node[name];
            node[name] = function (...args) { callback?.apply(this, args); update(); };
        }
        update();
    },
});
