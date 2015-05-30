all: thesis.pdf

thesis.pdf: thesis.tex thesis.bib
	pdflatex thesis.tex
	bibtex   thesis
	pdflatex thesis.tex
	pdflatex thesis.tex
