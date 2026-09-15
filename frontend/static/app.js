/* =================================================================
   SIH26166 — Adaptive Hybrid Lunar Image Registration
   Frontend JavaScript
   ================================================================= */

const form             = document.getElementById("registrationForm");
const runButton        = document.getElementById("runButton");
const statusEl         = document.getElementById("status");
const resultsCard      = document.getElementById("resultsCard");
const visualizationCard = document.getElementById("visualizationCard");
const acceptedBadge    = document.getElementById("acceptedBadge");
const imageNames       = document.getElementById("imageNames");
const resultsBody      = document.getElementById("resultsBody");
const inlierImage      = document.getElementById("inlierImage");

const imageAInput      = document.getElementById("imageA");
const imageBInput      = document.getElementById("imageB");
const dropzoneA        = document.getElementById("dropzoneA");
const dropzoneB        = document.getElementById("dropzoneB");
const hintA            = document.getElementById("hintA");
const hintB            = document.getElementById("hintB");
const previewA         = document.getElementById("previewA");
const previewB         = document.getElementById("previewB");


/* -----------------------------------------------------------------
   File Selection & Preview
   ----------------------------------------------------------------- */

function handleFileSelect(input, dropzone, hint, preview) {
    input.addEventListener("change", () => {
        const file = input.files[0];

        if (file) {
            dropzone.classList.add("has-file");
            hint.textContent = file.name;

            /* Show thumbnail preview */
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.hidden = false;
            };
            reader.readAsDataURL(file);
        } else {
            dropzone.classList.remove("has-file");
            hint.textContent = "Click to select or drag a file";
            preview.hidden = true;
            preview.removeAttribute("src");
        }
    });
}

handleFileSelect(imageAInput, dropzoneA, hintA, previewA);
handleFileSelect(imageBInput, dropzoneB, hintB, previewB);


/* -----------------------------------------------------------------
   Status Helpers
   ----------------------------------------------------------------- */

function setStatus(message, kind = "") {
    statusEl.textContent = message;
    statusEl.className = `status ${kind}`.trim();
}

function resetResults() {
    resultsCard.hidden = true;
    visualizationCard.hidden = true;
    resultsBody.innerHTML = "";
    inlierImage.removeAttribute("src");
}


/* -----------------------------------------------------------------
   Table Builder — Groups metrics by category
   ----------------------------------------------------------------- */

/**
 * Metric grouping for visual organisation in the results table.
 * Maps backend keys to display groups. Metrics not in any group
 * are placed in an "Other" section.
 */
const METRIC_GROUPS = [
    {
        label: "Pipeline",
        keys: [
            "accepted",
            "stage",
            "registration_mode",
        ],
    },
    {
        label: "Rotation",
        keys: [
            "rotation_used",
            "rotation_supported",
            "rotation_angle_deg",
            "rotation_confidence",
        ],
    },
    {
        label: "Seed Stage",
        keys: [
            "seed_inliers",
            "seed_coverage",
            "seed_area_ratio",
            "seed_anisotropy",
            "seed_sane",
        ],
    },
    {
        label: "Recovery & Refinement",
        keys: [
            "recovery_used",
            "recovery_matches",
            "lk_points",
        ],
    },
    {
        label: "Final Registration",
        keys: [
            "final_inliers",
            "final_rms",
            "final_coverage",
            "final_cells",
            "final_area_ratio",
            "final_anisotropy",
            "final_geometry_ok",
        ],
    },
    {
        label: "Timing",
        keys: [
            "seed_runtime_s",
            "lk_runtime_s",
            "rotation_runtime_s",
            "runtime_s",
        ],
    },
];

/**
 * Assign a row to a group based on its key.
 * Returns the group index (or a fallback "other" index).
 */
function getGroupIndex(key) {
    for (let i = 0; i < METRIC_GROUPS.length; i++) {
        if (METRIC_GROUPS[i].keys.includes(key)) return i;
    }
    return METRIC_GROUPS.length; // "Other"
}

/**
 * Build grouped table rows from the backend's table array.
 */
function buildResultsTable(tableData) {
    resultsBody.innerHTML = "";

    if (!tableData || tableData.length === 0) return;

    /* Sort rows into groups, preserving backend order within each group. */
    const groups = METRIC_GROUPS.map((g) => ({ label: g.label, rows: [] }));
    groups.push({ label: "Other", rows: [] });

    for (const row of tableData) {
        const idx = getGroupIndex(row.key);
        groups[idx].rows.push(row);
    }

    /* Render each non-empty group. */
    for (const group of groups) {
        if (group.rows.length === 0) continue;

        /* Group header */
        const headerTr = document.createElement("tr");
        headerTr.className = "table-group-header";
        const headerTd = document.createElement("td");
        headerTd.colSpan = 2;
        headerTd.textContent = group.label;
        headerTr.appendChild(headerTd);
        resultsBody.appendChild(headerTr);

        /* Data rows */
        for (const row of group.rows) {
            const tr = document.createElement("tr");

            /* Highlight the accepted row */
            if (row.key === "accepted") {
                tr.className = "row-accepted";
                const isTrue = row.raw === true || row.value === "True";
                tr.classList.add(isTrue ? "is-true" : "is-false");
            }

            const labelTd = document.createElement("td");
            labelTd.textContent = row.label;

            const valueTd = document.createElement("td");
            valueTd.textContent = row.value;

            tr.appendChild(labelTd);
            tr.appendChild(valueTd);
            resultsBody.appendChild(tr);
        }
    }
}


/* -----------------------------------------------------------------
   Form Submission
   ----------------------------------------------------------------- */

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const imageA = imageAInput.files[0];
    const imageB = imageBInput.files[0];

    if (!imageA || !imageB) {
        setStatus("Please select both lunar images.", "error");
        return;
    }

    resetResults();
    runButton.disabled = true;
    setStatus("Running adaptive registration…", "running");

    const formData = new FormData();
    formData.append("image_a", imageA);
    formData.append("image_b", imageB);

    try {
        const response = await fetch("/run", {
            method: "POST",
            body: formData
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(`Server returned HTTP ${response.status}.`);
        }

        if (!response.ok) {
            throw new Error(
                data.error || `Request failed (${response.status}).`
            );
        }

        /* -- Populate results ------------------------------------ */

        imageNames.textContent =
            `${data.image_a_name}  ↔  ${data.image_b_name}`;

        acceptedBadge.textContent =
            data.accepted ? "✓ ACCEPTED" : "✗ REJECTED";

        acceptedBadge.className =
            `badge ${data.accepted ? "good" : "bad"}`;

        /* Build the grouped metrics table. */
        buildResultsTable(data.table || []);

        resultsCard.hidden = false;

        /* -- Inlier visualization -------------------------------- */

        if (data.inlier_image) {
            inlierImage.src =
                `${data.inlier_image}?t=${Date.now()}`;

            visualizationCard.hidden = false;
        }

        setStatus(
            data.accepted
                ? "Registration accepted."
                : "Registration rejected by the validation gates.",
            data.accepted ? "success" : "error"
        );

    } catch (error) {
        resetResults();

        setStatus(
            error.message || "Registration failed.",
            "error"
        );

    } finally {
        runButton.disabled = false;
    }
});