set title "Simple demo of scatter data conversion to grid data"
unset hidden3d
set ticslevel 0.5
set view 60,30
set autoscale
set parametric
set style data points
set xlabel "data style point - no dgrid"
set key box
splot 'save_lecun_siamese2_iter100k.data' u 1:2:3:4 palette
