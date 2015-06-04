gnuplot <<EOF
set grid xtics ytics
set boxwidth 0.9 relative
set xlabel 'Iterations'

set terminal pngcairo font 'Droid Sans,9' size 500,300
set title "Loss"
set output "final_loss_3Dc3-vs-0c_iter300.png"
plot for [i=2:5] 'final_loss_3Dc3-vs-0c_iter300' u 1:i w linespoints title columnheader(i)
EOF
