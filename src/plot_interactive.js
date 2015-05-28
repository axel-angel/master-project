var chart;
var lastwheel = new Date();
var chart_focus = false;

window.onload = function () {
    on_hash_change();

  // handle zoom with mouse wheel
  $('#chartContainer').bind('wheel', function (e) {
    var dir = e.originalEvent.deltaY < 0 ? +1 : -1;
    if (e.originalEvent.deltaY == 0) return; // ignore horizontal
    e.preventDefault();

    var now = new Date();
    if (now - lastwheel < 1000) return; // don't call it too often
    zoom(dir);
    lastwheel = now;
  });

  function zoom(dir) {
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
  }

  $('#chartContainer')
    .mouseenter(function () { chart_focus = true; })
    .mouseleave(function () { chart_focus = false; });

  // handle move and zoom with keyboard arrows
  $('body').keydown(onkeypress);
  $('body').keypress(onkeypress);
  function onkeypress(e) {
    if (!chart_focus) return; // only when focusing
    if (e.charCode != 0) return; // avoid bug with ( in chrome

    var dirx = 0;
    var diry = 0;
    if (e.keyCode == 39) dirx = +1;
    if (e.keyCode == 37) dirx = -1;
    if (e.keyCode == 38) diry = +1;
    if (e.keyCode == 40) diry = -1;
    if (e.shiftKey && diry != 0) {
        zoom(diry);
        e.preventDefault();
    }
    else if (dirx != 0 || diry != 0) {
        move(dirx, diry);
        e.preventDefault();
    }
  }

  function move(dirx, diry) {
    var dx = chart.options.axisX.maximum - chart.options.axisX.minimum;
    var dy = chart.options.axisY.maximum - chart.options.axisY.minimum;
    chart.options.axisX.minimum += dx * dirx / 4;
    chart.options.axisY.minimum += dy * diry / 4;
    chart.options.axisX.maximum += dx * dirx / 4;
    chart.options.axisY.maximum += dy * diry / 4;
    chart.render();
  }
};

$(window).on('hashchange', on_hash_change);

function on_hash_change() {
  // load js specified after the hash
  var $debug = $('#debug');
  $debug.text("Loading...");
  if (chart) {
      chart.options.data = [];
      chart.render();
  }
  var data_url = window.location.hash.substr(1);
  $.get(data_url, function (j) {
      eval(j);
      plot();
      $debug.text("Data loaded: "+ data_url);
  }).error(function (e) {
      console.error(e);
      $debug.text("Error: "+ e.statusText);
  });
}

function plot() {
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
    axisX: { minimum: -6, maximum: 6, gridThickness: 1 },
    axisY: { minimum: -6, maximum: 6, gridThickness: 1 },
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
        markerSize: 1,
        legendText: legends[k],
        showInLegend: "true",
        markerType: "circle",
        dataPoints: data[k],
      };
    }).concat([{
        type: "scatter",
        markerSize: 10,
        legendText: 'Overlay',
        showInLegend: "true",
        markerType: "triangle",
        dataPoints: [],
        markerColor: 'yellow',
        markerBorderThickness: 1,
        markerBorderColor: 'black',
    }]),
  });
  chart.render();
}

function applyFilters() {
  var $form = $('#filters');
  var label = $form.find('input[name="label"]').val();
  var src = $form.find('input[name="src"]').val();
  var sample = $form.find('input[name="sample"]').val();
  var label_rx = new RegExp(label, 'i');
  var src_rx = new RegExp(src, 'i');
  var sample_rx = new RegExp(sample, 'i');

  var overlayData = chart.options.data.filter(function (d) {
    return (d.legendText == "Overlay");
  })[0];

  chart.options.data.forEach(function (d) {
    if (d == overlayData) return;
    var first = d.dataPoints[0];
    d.visible = (label_rx.test(first.l))
             && (src_rx.test(first.src));
  });

  overlayData.dataPoints.length = 0;
  overlayPoints = []
  if (sample.length > 0) {
    chart.options.data.forEach(function (d) {
      d.dataPoints.forEach(function (p) {
        if (sample_rx.test(p.sample)) {
          overlayPoints.push(p)
        }
      });
    });
    overlayData.dataPoints = overlayPoints;
  }
  chart.render();
}
