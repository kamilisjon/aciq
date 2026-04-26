TECTONIC ?= tectonic
TEX := derivation.tex
PDF := $(TEX:.tex=.pdf)

.PHONY: pdf clean

pdf: $(PDF)

$(PDF): $(TEX)
	$(TECTONIC) -X compile $(TEX)

clean:
	rm -f $(PDF) *.aux *.log *.out
