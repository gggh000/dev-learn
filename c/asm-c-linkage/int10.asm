    global    _start

            section   .text
_start:     mov     al, 'A'                 ; char 2 display.
            mov     ah, 0x10                ; int 10h, write char.
            sub     bh, bh                  ; page to write.
            mov     cx, 0x10                ; No. of times to write to screen.
            int     0x10
	    jmp     $

            section   .data
