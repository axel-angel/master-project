set xtics rotate by -45 scale 0
set grid ytics
set boxwidth 0.9 relative
set style data histogram
set style histogram cluster errorbars gap 1 lw 1
set style fill solid 1.0 border lt -1
set bars front

set terminal pngcairo font 'Droid Sans,9' size 3200,600
set title "Accuracy % of testset for translation (x+y)"
set output 'test_transfo_v6_v3-shift3b_translate.png'
plot for [COL=2:16:2] 'test_transfo_v6_v3-shift3b_translate.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 600,600
set output 'test_transfo_v6_v3-shift3b_translate_avg.png'
plot for [COL=2:16:2] 'test_transfo_v6_v3-shift3b_translate_avg.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 1200,600
set title "Accuracy % of testset for rotation"
set output 'test_transfo_v6_v3-shift3b_rotate.png'
plot for [COL=2:8:2] 'test_transfo_v6_v3-shift3b_rotate.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 600,600
set output 'test_transfo_v6_v3-shift3b_rotate_avg.png'
plot for [COL=2:8:2] 'test_transfo_v6_v3-shift3b_rotate_avg.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 1200,600
set title "Accuracy % of testset for translation (xy grid)"
set output 'test_transfo_translate-v2.png'
plot for [COL=2:16:2] 'test_transfo_translate-v2.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 1200,600
set title "Accuracy % of testset for rotation (r grid)"
set output 'test_transfo_rotate-v2.png'
plot for [COL=2:16:2] 'test_transfo_rotate-v2.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 1200,600
set title "Classified like adversial target label % (adversial for i)"
set output 'test_transfo_adversial.png'
plot for [COL=2:16:2] 'test_transfo_adversial.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 1200,600
set title "Accuracy % (adversial for i)"
set output 'test_transfo_adversial_correct.png'
plot for [COL=2:16:2] 'test_transfo_adversial_correct.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 1200,600
set title "Accuracy % of testset for translation (xy random)"
set output 'test_transfo_translate-v3.png'
plot for [COL=2:16:2] 'test_transfo_translate-v3.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)

set terminal pngcairo font 'Droid Sans,9' size 1200,600
set title "Accuracy % of testset for rotate (r random)"
set output 'test_transfo_rotate-v3.png'
plot for [COL=2:16:2] 'test_transfo_rotate-v3.data' using (abs(column(COL)*100)):(sqrt(100*column(COL+1))):xticlabels(1) title columnheader(COL)
