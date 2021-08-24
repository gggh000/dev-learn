https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
4.6 PHONY TARGETS

clean1 file exists.
Therefore clean1 target does not execute rm command.

clean file exists too.
Clean target does execute because it is declared as .PHONY in Makefile.


