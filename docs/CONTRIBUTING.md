# Contributing to SleepLearning

## Coding style

Most important rules:

* We use **python3** in this project
* 4 spaces for indentation rather than tabs
* 80 character line length
* PEP8 formatting

When unsure find more info on the following links:

1. [Google style](https://google.github.io/styleguide/pyguide.html)
2. [PEP](https://www.python.org/dev/peps/pep-0008/)
3. [Hitchhiker's guide](http://docs.python-guide.org/en/latest/writing/structure/)

**Pay special attention to code documentation!**

##  Reporting

- Intermediate reports and exploratory data analysis are done using [Jupyter](http://jupyter.org/). Nice article to get started can be found [here](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook).
All useful plots should be generated within notebooks. Notebooks are meant to use core functionalities of sleeplearning package.


##  Project management

- Project management, task distribution, information on current problems and findings, issues and ideas should be tracked using **TOADD**.

- For an independent method exploration and algorithm development one can optionally design and test algorithms separately using [git branching](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) and then only later can integrate with the main branch.

## Literature research and related work

For all relevant literature, papers, available code, anything useful for the comparison, modeling or ideas, use the following **TOADD**.
If possible, add short explanations why you think the chosen paper is useful for the selected application (alignment, normalization, classification etc.). 

## Testing

Important packages for testing are:

1. [unittest](https://docs.python.org/3/library/unittest.html)
2. [doctest](https://docs.python.org/3/library/doctest.html#module-doctest)

The following links might be useful
- [Hitchhiker's guide](http://docs.python-guide.org/en/latest/writing/tests/)

## New files and scripts

When adding new files and scripts, please respect the existing folder hierarchy which is described bellow:

[+] data\
[+] bin\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-] classify.py\
[+] sleeplearning\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[+] lib\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[+] tests\
[+] reports\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-] sleep-learning-analysis.ipynb\
[+] docs\
[+] web\
[-] README.md\
[-] LICENSE\
[-] .gitignore\
[-] requirements.txt

**data/** contains several sample inputs

**bin/** contains executable scripts.

* *classify.py* is used for sleep phase classification

**sleeplearning/** contains the core part of the project

* **lib/** provides all necessary algorithms and data structures for sleeplearning data processing
* **tests/** contains scripts for unit testing of individual code modules

**reports/** contains reports on individual functionalities of the computational model or the exploratory analysis of MS data. It can contatin descriptions, useful plots, statistics etc. The goal is to document scientific achievements and make it easier to introduce colleagues and collaborators to new findings and progress. The notebooks are written using [Jupyter](http://jupyter.org/). Other possibilities are word and latex documents, slides etc. (A good idea is to keep the editable version of the document e.g. .tex instead of .pdf so we can add changes when necessary)

**docs/** contains necessary documentation on the project and the repository

**web/** contains tools and description of ways to deploy the model on-line as a web server

**requirements.txt** contains necessary installation modules

This structure is partially inspired by [link](https://coderwall.com/p/lt2kew/python-creating-your-project-structure).
