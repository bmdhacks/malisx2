/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Link-time layout checks between builds of the same structure.
 *
 * The command-stream back half is compiled twice (PAN_ARCH 10 and 11), and
 * code built once without a PAN_ARCH hands it the same objects. A struct
 * that came out different in two of those builds would be read at the
 * wrong offsets, silently. To catch that when the driver is linked, one
 * build defines a symbol whose name carries the value (a size or an
 * offset) and the other refers to the symbol named after its own value:
 *
 *    MALI_LAYOUT_DEFINE(cmd_buffer_size, sizeof(struct mali_cmd_buffer));
 *    MALI_LAYOUT_EXPECT(cmd_buffer_size, sizeof(struct mali_cmd_buffer));
 *
 * If the two values differ, the reference names a symbol nobody defines
 * and the link fails with "undefined reference to
 * mali_layout_cmd_buffer_size_<value>". The symbols are hidden and
 * absolute (value 0); the reference is one pointer-sized word of
 * read-only data. Each macro is used inside a function body, where the
 * compiler can print the constant into the assembler text.
 */

#ifndef MALI_LAYOUT_CHECK_H
#define MALI_LAYOUT_CHECK_H

#define MALI_LAYOUT_DEFINE(name, value)                                        \
   __asm__ volatile(".globl mali_layout_" #name "_%c0\n\t"                    \
                    ".hidden mali_layout_" #name "_%c0\n\t"                   \
                    ".set mali_layout_" #name "_%c0, 0" ::"i"(value))

#define MALI_LAYOUT_EXPECT(name, value)                                        \
   __asm__ volatile(".pushsection .rodata.mali_layout, \"a\"\n\t"            \
                    ".hidden mali_layout_" #name "_%c0\n\t"                   \
                    ".balign 8\n\t"                                            \
                    ".quad mali_layout_" #name "_%c0\n\t"                     \
                    ".popsection" ::"i"(value))

#endif
