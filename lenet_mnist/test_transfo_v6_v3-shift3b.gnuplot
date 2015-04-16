set xtics rotate by -45 scale 0
set grid ytics
set boxwidth 0.9 relative
set style data histogram
set style histogram cluster errorbars gap 1 lw 1
set style fill solid 1.0 border lt -1
set bars front
set terminal pngcairo font 'Droid Sans,9' size 1200,600

set title "Accuracy % of testset for translation"
set output 'test_transfo_v6_v3-shift3b_translate.png'
plot for [COL=2:6:2] 'test_transfo_v6_v3-shift3b_translate.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set title "Accuracy % of testset for rotation"
set output 'test_transfo_v6_v3-shift3b_rotate.png'
plot for [COL=2:4:2] 'test_transfo_v6_v3-shift3b_rotate.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)
