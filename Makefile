# create and configure environment. If it already exists, update it with environment.yml
.PHONY: env
env:
	@if conda env list | grep -q "stat159-final_group02"; then \
		conda env update -n stat159-final_group02 -f environment.yml; \
	else \
		conda env create -f environment.yml; \
	fi

# build the html rendering of the MyST site
.PHONY: html
html:
	myst build --html

# clean up the output, pdfs, and _build folders
.PHONY: clean
clean:
	rm -rf Output/*
	rm -rf pdf_builds/*
	rm -rf _build/*