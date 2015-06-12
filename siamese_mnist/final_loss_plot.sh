#!/bin/sh

gnuplot <<EOF
set grid xtics ytics
set boxwidth 0.9 relative
set xlabel 'Iterations'

set terminal pngcairo font 'Droid Sans,9' size 500,300
set title "Loss on LeCun MNIST paired dataset"
set output "final_loss_test2bv7.png"
plot 'final_loss_test2bv7_2bv7' u 1:2 w linespoints title "DrLIM", \
     'final_loss_test2bv7_3Dc3' u 1:2 w linespoints title "Ours"
EOF
