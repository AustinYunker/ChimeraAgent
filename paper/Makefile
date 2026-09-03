# Build the LNCS manuscript. No latexmk on this host, so the passes are explicit.
#
#   make          -- build main.pdf
#   make pages    -- build, then report the page count (limit: 6 excluding refs)
#   make clean    -- remove build artefacts, keep the fetched class

TEX  := pdflatex -interaction=nonstopmode -halt-on-error
MAIN := main

.PHONY: all pages clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib llncs.cls
	$(TEX) $(MAIN)
	-bibtex $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

llncs.cls:
	./fetch-template.sh

pages: $(MAIN).pdf
	@pdfinfo $(MAIN).pdf | awk '/^Pages/ {print "Total pages: " $$2 "  (limit is 6 excluding references)"}'

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf

# Also drop the fetched Springer class.
distclean: clean
	rm -f llncs.cls splncs04.bst
