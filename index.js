const main = document.querySelector('#main');
const jsonName = ["mark.json", "vbar.json", "hit_map.json", "sankey.json"];
const select = document.querySelector('#select');


/**
 * ajoute tout les nom dans jsonName a la select box 
 */
function addToSelect(){
    jsonName.forEach(function(name) {
        const option = document.createElement('option');
        option.value = name;
        option.innerHTML = name;
        select.appendChild(option);
    });
}


/**
 * fonction qui crée le graphique dans la div 
 * @param {url du json} json 
 */
function addChart(json) {
    var spec = json;
    var embedOpt = {"mode": "vega-lite"};

    function showError(el, error){
        document.getElementById("content-chart").innerHTML = ('<div class="error" style="color:red;">'
                        + '<p>JavaScript Error: ' + error.message + '</p>'
                        + "<p>This usually means there's a typo in your chart specification. "
                        + "See the javascript console for the full traceback.</p>"
                        + '</div>');
        throw error;
    }
    vegaEmbed("#content-chart", spec, embedOpt)
      .catch(error => showError(document.getElementById("content-chart"), error));
}


select.addEventListener('change', (event) => {
    addChart("http://127.0.0.1:5500/json_chart/" + event.target.value);
});

addToSelect();
addChart("http://127.0.0.1:5500/json_chart/" + jsonName[0]);

