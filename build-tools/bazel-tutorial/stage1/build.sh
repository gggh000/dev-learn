bazel build //main:hello-world
bazel query --notool_deps --noimplicit_deps "deps(//main:hello-world)"   --output graph
sudo apt update && sudo apt install graphviz xdot
xdot <(bazel query --notool_deps --noimplicit_deps "deps(//main:hello-world)" \

