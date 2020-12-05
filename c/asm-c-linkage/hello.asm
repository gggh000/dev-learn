;   https://cs.lmu.edu/~ray/notes/nasmtutorial/
extern      functionC

    global    _start

            section   .text
_start:     mov     rax, 1                  ; system call for write
            mov     rdi, 1                  ; file handle 1 is stdout
            mov     rsi, message            ; address of string to output
            mov     rdx, 13                 ; number of bytes
            syscall                         ; invoke operating system to do the write

            mov edi,7                       ; pass a function parameter
            call otherFunction              ; run the function below, until it does "ret"
            add eax,1                       ; modify returned value

            push    rbp            
            call    functionC               ; call function c
            pop     rbp

            mov     rax, 60                 ; system call for exit
            xor     rdi, rdi                ; exit code 0
            syscall                         ; invoke operating system to exit

; a function "declaration" in assembly

otherFunction: 
            mov     eax, edi ; return our function's only argument
            ret

            section   .data
message:    db        "Hello, World", 10    ; note the newline at the end

