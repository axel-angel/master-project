#!/bin/sh

gnuplot <<EOF
set grid xtics ytics
set boxwidth 0.9 relative
set xlabel 'Iterations'
set terminal pngcairo font 'Droid Sans,9' size 500,300

set title "Loss on DrLIM MNIST paired test set"
set output "final_loss_test2bv7.png"
plot 'final_loss_test2bv7_2bv7' u 1:2 w linespoints title "DrLIM", \
     'final_loss_test2bv7_3Dc3' u 1:2 w linespoints title "Ours"


set title "Loss of our model on its test set (MNIST)"
set output "final_loss_testset_3Dc3.png"
plot for [i=2:4:1] 'final_loss_testset_3Dc3_iter500' u 1:i w linespoints title columnheader(i)

set yrange [0:0.5]
set title "Loss of DrLIM on its test set (MNIST)"
set output "final_loss_testset_2bv7.png"
plot for [i=2:2:1] 'final_loss_testset_2bv7_iter500' u 1:i w linespoints title columnheader(i)
EOF
