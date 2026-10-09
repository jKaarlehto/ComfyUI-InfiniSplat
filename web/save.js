import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "InfiniSplat.SaveControls",
    nodeCreated(node) {
        if (node.comfyClass !== "InfiniSplatGenerate") return;
        const save = node.widgets?.find(w => w.name === "save");
        const path = node.widgets?.find(w => w.name === "file_path");
        if (!save || !path) return;
        const originalType = path.type;
        const originalSize = path.computeSize;
        const update = () => {
            path.type = save.value ? originalType : "infinisplat_hidden";
            path.computeSize = save.value ? originalSize : () => [0, -4];
            const slot = node.outputs?.findIndex(output => output.name === "file_path") ?? -1;
            if (save.value && slot < 0) node.addOutput("file_path", "STRING", { shape: 6 });
            if (!save.value && slot >= 0) node.removeOutput(slot);
            node.setSize([node.size[0], node.computeSize()[1]]);
            app.graph.setDirtyCanvas(true, true);
        };
        const callback = save.callback;
        save.callback = function (...args) {
            callback?.apply(this, args);
            update();
        };
        const configure = node.onConfigure;
        node.onConfigure = function (...args) {
            configure?.apply(this, args);
            update();
        };
        update();
    },
});
