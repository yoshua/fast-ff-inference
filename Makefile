PAPER = fast-ff-inference

${PAPER}.pdf: ${PAPER}.tex  *.bib
	pdflatex ${PAPER}.tex  
	bibtex ${PAPER}
	pdflatex ${PAPER}.tex 
	pdflatex ${PAPER}.tex

clean:
	rm -f ${PAPER}.aux ${PAPER}.blg ${PAPER}.pdf ${PAPER}.bbl ${PAPER}.log ${PAPER}.out ${PAPER}.spl


