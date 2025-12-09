all: main.pdf

main.pdf: main.tex
	pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

clean:
	rm -f main.aux main.log main.nav main.out main.snm main.toc main.pdf

view: main.pdf
	open main.pdf

.PHONY: all clean view
