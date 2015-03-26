window.onload = function () {
  // init our objects
  var data = {};
  var legends = {};
  Object.keys(label_set).forEach(function (l) {
    Object.keys(src_set).forEach(function (src) {
      var k = l +"_"+ src;
      data[k] = [];
    });
  });

  // for each point, assign it to one group (based on key)
  X.forEach(function (x) {
    var l = x.l;
    var src = x.src
    var k = l +"_"+ src;
    x.label = "<b>"+ l +"</b> "
      + "(#"+ x.i +") "
      + "["+ x.src +" | "+ x.tr +"("+ x.v +")]<br/>"
      + "<img class='digit' src='data:image/jpeg;base64,"+ imgs[x.i] +"'/>";
    data[k].push(x);
    legends[k] = x.l +" "+ src;
  });

  // remove empty keys
  Object.keys(data).forEach(function (k) {
    if (data[k].length == 0)
      delete data[k];
  });

  // config the plot and render it
  var data_keys = Object.keys(data);
  data_keys.sort();
  chart = new CanvasJS.Chart("chartContainer", {
    axisX: { minimum: -6, maximum: 6 },
    axisY: { minimum: -6, maximum: 6 },
    legend: { // TODO: remove because our filters are better!
      cursor: "pointer",
      itemclick: function (e) { // toggle cluster when click on legend
        e.dataSeries.visible = !(typeof(e.dataSeries.visible) === "undefined"
                             || e.dataSeries.visible)
        e.chart.render();
      }
    },
    data: data_keys.map(function (k) {
      return {
        type: "scatter",
        markerSize: 2,
        legendText: legends[k],
        showInLegend: "true",
        markerType: "circle",
        dataPoints: data[k],
      };
    }),
  });
  chart.render();

  // handle zoom with mouse wheel
  var lastwheel = new Date();
  $('#chartContainer').bind('wheel', function (e) {
    e.preventDefault();
    var now = new Date();
    if (now - lastwheel < 1000) return; // don't call it too often
    if (e.originalEvent.deltaY == 0) return; // ignore horizontal

    var dir = e.originalEvent.deltaY < 0 ? +1 : -1;
    var scale = dir > 0 ? 1/2 : 2;
    var subscale = dir > 0 ? +1/4 : -1/2;

    chart.options.data.forEach(function (d) {
      d.markerSize /= Math.sqrt(scale); // heuristic to avoid overlap, works!
    });
    var dx = chart.options.axisX.maximum - chart.options.axisX.minimum;
    var dy = chart.options.axisY.maximum - chart.options.axisY.minimum;
    chart.options.axisX.minimum += dx * subscale;
    chart.options.axisY.minimum += dy * subscale;
    chart.options.axisX.maximum -= dx * subscale;
    chart.options.axisY.maximum -= dy * subscale;
    chart.render()
    lastwheel = now;
  });

  // handle move with keyboard arrows
  $('body').keydown(onkeypress);
  $('body').keypress(onkeypress);
  function onkeypress(e) {
    var dirx = 0;
    var diry = 0;
    if (e.keyCode == 39) dirx = +1;
    if (e.keyCode == 37) dirx = -1;
    if (e.keyCode == 38) diry = +1;
    if (e.keyCode == 40) diry = -1;

    if (dirx == 0 && diry == 0) return;
    var dx = chart.options.axisX.maximum - chart.options.axisX.minimum;
    var dy = chart.options.axisY.maximum - chart.options.axisY.minimum;
    chart.options.axisX.minimum += dx * dirx / 4;
    chart.options.axisY.minimum += dy * diry / 4;
    chart.options.axisX.maximum += dx * dirx / 4;
    chart.options.axisY.maximum += dy * diry / 4;
    chart.render();

    e.preventDefault();
  };
}
