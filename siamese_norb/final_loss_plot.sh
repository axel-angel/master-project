#!/bin/sh

gnuplot <<EOF
set grid xtics ytics
set boxwidth 0.9 relative
set xlabel 'Iterations'
set terminal pngcairo font 'Droid Sans,9' size 500,300

#set title "Loss on LeCun NORB paired dataset"
#set output "final_loss_test2bv7.png"
#plot 'final_loss_test2bv7_2bv7' u 1:2 w linespoints title "DrLIM", \
#     'final_loss_test2bv7_3Dc3' u 1:2 w linespoints title "Ours"


set title "Loss of our model on its testset (NORB)"
set yrange [0:3]
set xrange [0:10000]
set output "final_loss_testset_2D1g2.png"
plot for [i=2:4:1] 'final_loss_testset_2D1g2_iter50k' u 1:i w linespoints title columnheader(i)

set title "Loss of DrLIM on its testset (NORB)"
set output "final_loss_testset_3d.png"
plot for [i=2:2:1] 'final_loss_testset_3d_iter50k' u 1:i w linespoints title columnheader(i)
EOF
