    global    _start

            section   .text
_start:     
;	set video mode
	mov	ax, 0x0002
	int	0x10

;	write 'A' 16 times at current cursor.
        mov     ah, 0x0e                ; int 10h, write char.
	mov 	al, '#'                 ; char 2 display.
        int     0x10
        mov     ah, 0x0e                ; int 10h, write char.
	mov 	al, '$'                 ; char 2 display.
        int     0x10

	mov	ah, 0x42		; bios 13h extended read service code.
	mov	dl, 0x81		; drive No.

;	DS:SI - pointer to DAP (disk access packet).

	mov	ax, 0x7c00
	mov	ds, ax
	lea	si, [DAP]

	int 	0x13			; issue the command.
	
	mov	ax, 0x8000
	push 	ax
	ret

DAP:
;	DAP packet for bios int 13h (ah=0x42)
	db 	0x10			; size of this data struct.
	db 	0x00			; unused.
	dw	0x02			; No. of sectors to read.
	dd	0x00008000		; segment:offset of target location in memory
	dd	0x28			; not sure this needs to be inspected using ext2 on hdd not fdd.

        section   .data
