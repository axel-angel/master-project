set title "Simple demo of scatter data conversion to grid data"
unset hidden3d
set ticslevel 0.5
set view 60,30
set autoscale
set parametric
set style data points
set xlabel "data style point - no dgrid"
set key box
splot "save_lecun_siamese0b_iter1k.data"
