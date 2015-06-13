var chart;
var _chart;
var lastwheel = new Date();
var chart_focus = false;
var tooltips_map = {};

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
    chart_render();
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
    chart_render();
  }

  $('#filters').submit(applyFilters);
};

$(window).on('hashchange', on_hash_change);

function on_hash_change() {
  // load js specified after the hash
  var $debug = $('#debug');
  $debug.text("Loading...");
  if (chart) {
      chart.options.data = [];
      chart_render();
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
    data[k].push(x);
    legends[k] = x.l +" "+ src;
  });

  // remove empty keys
  Object.keys(data).forEach(function (k) {
    if (data[k].length == 0)
      delete data[k];
  });

  var point_click = function (e) {
    var idx1 = e.dataSeriesIndex;
    var idx2 = e.dataPointIndex;
    var k = idx1 +"_"+ idx2;
    var t = tooltips_map[k];
    var x = e.dataPoint;
    //console.log(['click', e]);
    if (typeof(t) == "undefined") {
      t = new ToolTip(_chart, _chart._options.toolTip, _chart.theme);
      t.getToolTipInnerHTML = function () {
        return "<img class='digit' src='data:image/jpeg;base64,"+ imgs[x.i] +"'/>";
      };
      $(t.contentDiv).parent().attr('class', 'mytooltip');
      t._updateToolTip(e.x, e.y);
      tooltips_map[k] = t;
    }
    else {
      t.hide(false);
      delete tooltips_map[k];
    }
  };

  // config the plot and render it
  var data_keys = Object.keys(data);
  data_keys.sort();
  chart = new CanvasJS.Chart("chartContainer", {
    axisX: { minimum: -6, maximum: 6, gridThickness: 1 },
    axisY: { minimum: -6, maximum: 6, gridThickness: 1 },
    legend: { // TODO: remove because our filters are better!
      fontSize: 15,
      cursor: "pointer",
      itemclick: function (e) { // toggle cluster when click on legend
        e.dataSeries.visible = !(typeof(e.dataSeries.visible) === "undefined"
                             || e.dataSeries.visible)
        chart_render();
      }
    },
    toolTip: {
      content: function (e) {
        var x = e.entries[0].dataPoint;
        var l = x.l;
        var src = x.src
        var img = imgs[x.i];
        return "<b>"+ l +"</b> "
          + "(#"+ x.i +")<br/>"
          + "["+ x.src +" | "+ x.tr +"("+ x.v +")]<br/>"
          + "<img class='digit' src='data:image/jpeg;base64,"+ img +"'/>";
      },
    },
    data: data_keys.map(function (k) {
      return {
        type: "scatter",
        markerSize: 2,
        legendText: legends[k],
        showInLegend: "true",
        markerType: "circle",
        dataPoints: data[k],
        click: point_click,
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
        click: point_click,
    }]),
  });
  _chart = chart._internal;
  chart_render();
}

function applyFilters(e) {
  if (e) e.preventDefault();
  var $form = $('#filters');
  var label = $form.find('input[name="label"]').val();
  var label_rx = new RegExp(label, 'i');
  var src = $form.find('input[name="src"]').val();
  var src_rx = new RegExp(src, 'i');
  var sample = $form.find('input[name="sample"]').val();
  var sample_rx = new RegExp(sample, 'i');
  var transfo = $form.find('input[name="transfo"]').val();
  var transfo_rx = new RegExp(transfo, 'i');

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
  if (sample.length > 0 || transfo.length > 0) {
    chart.options.data.forEach(function (d) {
      d.dataPoints.forEach(function (p) {
        var cond = ( sample.length == 0  || sample_rx.test(p.sample) )
                && ( transfo.length == 0 || transfo_rx.test(p.v) )
        if (cond) {
          overlayPoints.push(p)
        }
      });
    });
    overlayData.dataPoints = overlayPoints;
  }
  chart_render();
}

function chart_render() {
  chart.render();

  for (var k in tooltips_map) {
    tooltips_map[k].hide();
    delete tooltips_map[k];
  }
}
