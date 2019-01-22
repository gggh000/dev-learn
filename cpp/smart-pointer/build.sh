UNIQUE_PTR=unique-ptr
SHARED_PTR=shared-ptr

FILES=( \
$UNIQUE_PTR \
$SHARED_PTR \
)

for (( i = 0 ; i < ${#FILES[@]} ; i ++ ))
do
    echo building ${FILES[$i]} ...
    g++ ${FILES[$i]}.cpp  -std=c++0x -o ${FILES[$i]}.out
done
