nasm -felf64 -F dwarf int10.asm
ld int10.o
echo "use gdb a.out to start debugging"
tail -c $((1680-0x80)) a.out | head -c 446 > a1.out
echo "code portion of a.out is written to a1.out."
echo "use hexdump -C <filename> -n <No. of bytes> to display it."

