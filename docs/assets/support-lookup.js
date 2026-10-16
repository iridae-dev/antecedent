(() => {
  const panel = document.getElementById("support-lookup");
  if (!panel) return;
  const source = new URL("support-matrix.json", document.currentScript.src);
  const status = document.getElementById("support-count");
  fetch(source).then(response => {
    if (!response.ok) throw new Error("Registry unavailable");
    return response.json();
  }).then(registry => {
    const rows = registry.licensed;
    const fields = ["query", "graph_class", "structure", "inference", "validation"];
    const labels = ["Query", "Graph class", "Structure source", "Inference", "Validation"];
    const filters = fields.map((field, index) => {
      const label = document.createElement("label");
      label.textContent = labels[index] + " ";
      const select = document.createElement("select");
      select.add(new Option("All", ""));
      [...new Set(rows.map(row => row[field]))].sort().forEach(value => select.add(new Option(value, value)));
      label.append(select);
      document.getElementById("support-filters").append(label);
      return select;
    });
    const search = document.getElementById("support-search");
    const render = () => {
      const query = search.value.trim().toLowerCase();
      const matches = rows.filter(row => fields.every((field, index) => !filters[index].value || row[field] === filters[index].value) && JSON.stringify(row).toLowerCase().includes(query));
      const body = document.getElementById("support-rows");
      body.replaceChildren();
      matches.forEach(row => {
        const tr = document.createElement("tr");
        [...fields.map(field => row[field]), (row.estimators || [row.route_estimator]).join(", ")].forEach(value => {
          const td = document.createElement("td");
          td.textContent = value;
          tr.append(td);
        });
        body.append(tr);
      });
      status.textContent = `${matches.length} of ${rows.length} licensed cells match. Calibration and empirical support require their own checks.`;
    };
    filters.forEach(select => select.addEventListener("change", render));
    search.addEventListener("input", render);
    render();
  }).catch(() => { status.textContent = "Could not load the registry. Use the full support matrix below."; });
})();
