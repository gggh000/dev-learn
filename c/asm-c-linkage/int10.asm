    global    _start

            section   .text
_start:     
;	set video mode
	mov	ax, 0x000d
	int	0x10

; 	set cursor
	mov 	ah, 0x2  		; set cursor function.
	sub 	bh, bh			; select page.
	mov 	dl, 10			; row 10.
	sub	dh, dh			; col 0.
	int 	0x10			; call

;	write 'A' 16 times at current cursor.
	mov 	al, 'A'                 ; char 2 display.
        mov     ah, 0x10                ; int 10h, write char.
        sub     bh, bh                  ; page to write.
	sub     bl, bl		    	; attrib: text mode.
        mov     cx, 0x10                ; No. of times to write to screen.
        int     0x10
	jmp     $

        section   .data
