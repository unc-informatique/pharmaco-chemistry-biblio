const diagrams = ["vbar.json", "sankey.json", "mark.json"];

function showError(el, error) {
  el.innerHTML = `
    <div class="error" style="color:red;">
    <p>JavaScript Error: ${error.message} </p>
    <p>This usually means there's a typo in your chart specification.
       See the javascript console for the full traceback
    </p>
    </div>
    `;
  throw error;
}

/**
 * fonction qui crée le graphique dans la div
 * @param {url du json} jsonPath
 */
function addChart(jsonPath) {
  const opts = { };
  const container = document.createElement("div");

  container.classList.add("chart");
  document.querySelector("#content-chart").appendChild(container);

  vegaEmbed(container, jsonPath, opts).catch((error) => showError(container, error));
}

console.info(`vega.version = ${vega.version}`);
console.info(`vegaLite.version = ${vegaLite.version}`);
console.info(`vegaEmbed.version = ${vegaEmbed.version}`);

diagrams.forEach(function (name) {
  addChart(`charts/${name}`);
});
