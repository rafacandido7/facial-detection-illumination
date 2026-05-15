TEX := main.tex
PDF := $(TEX:.tex=.pdf)
LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdf watch clean distclean

all: pdf

pdf:
	rm -f $(PDF)
	$(LATEXMK) $(LATEXMK_FLAGS) $(TEX)
	$(LATEXMK) -c $(TEX)

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(TEX)

clean:
	$(LATEXMK) -c $(TEX)

distclean:
	$(LATEXMK) -C $(TEX)
