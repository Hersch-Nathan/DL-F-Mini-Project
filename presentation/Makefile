.PHONY: all clean view

all:
	pdflatex -interaction=nonstopmode herschnathanminiproject.tex > /dev/null
	pdflatex -interaction=nonstopmode herschnathanminiproject.tex > /dev/null

clean:
	rm -f herschnathanminiproject.aux herschnathanminiproject.log herschnathanminiproject.out herschnathanminiproject.toc herschnathanminiproject.nav herschnathanminiproject.snm herschnathanminiproject.fls herschnathanminiproject.fdb_latexmk herschnathanminiproject.synctex.gz herschnathanminiproject.vrb

view: all
	open herschnathanminiproject.pdf
