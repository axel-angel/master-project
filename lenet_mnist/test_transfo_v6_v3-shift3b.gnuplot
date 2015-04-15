set xtics rotate by -45
set title "Translation invariance"
set boxwidth 0.9 relative
set style data histogram
set style histogram cluster errorbars gap 0.5 lw 1
set style fill solid 1.0 border lt -1
set bars front
set terminal pngcairo font 'Droid Sans,9' size 800,600
set output 'test_transfo_v6_v3-shift3b.png'
plot for [COL=2:4:2] 'test_transfo_v6_v3-shift3b.data' using (abs(column(COL))):(sqrt(column(COL+1))):xticlabels(1) title columnheader(COL)
