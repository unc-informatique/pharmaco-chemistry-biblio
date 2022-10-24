const main = document.querySelector("#main");
const jsonName = ["vbar.json", "mark.json", "sankey_template.json"];

/**
 * ajoute tout les nom dans jsonName a la select box
 */
function addToSelect() {
  jsonName.forEach(function (name) {
    addChart(name, "json_chart/" + name);
  });
}

/**
 * fonction qui crée le graphique dans la div
 * @param {url du json} json
 */
function addChart(name, json) {
  var spec = json;
  var embedOpt = { mode: "vega-lite" };

  var div = document.createElement("li");
  var br = document.createElement("br");
  function showError(el, error) {
    div.innerHTML =
      '<div class="error" style="color:red;">' +
      "<p>JavaScript Error: " +
      error.message +
      "</p>" +
      "<p>This usually means there's a typo in your chart specification. " +
      "See the javascript console for the full traceback.</p>" +
      "</div>";
    throw error;
  }
  // create a new div

  // add the id name to the div but remove the .json
  div.classList.add("chart");
  name = name.replace(".json", "");
  div.id = name;
  // add the div to the main div
  document.querySelector("#content-chart").appendChild(br);
  document.querySelector("#content-chart").appendChild(div);

  vegaEmbed("#" + name, spec, embedOpt).catch((error) => showError(div, error));
}

console.info(`vega.version = ${vega.version}`);
console.info(`vegaLite.version = ${vegaLite.version}`);
console.info(`vegaEmbed.version = ${vegaEmbed.version}`);

addToSelect();
