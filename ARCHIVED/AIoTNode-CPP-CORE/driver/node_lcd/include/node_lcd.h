/**
 * @file node_lcd.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief LCD driver header file
 * @version 1.0
 * @date 2025-10-22
 * @ref Alinetek LCD driver
 * @copyright Copyright (c) 2024
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "node_spi.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Pin definitions */
#define LCD_NUM_BL GPIO_NUM_41  /* Backlight control pin */
#define LCD_NUM_WR GPIO_NUM_40  /* Write control pin */
#define LCD_NUM_CS GPIO_NUM_39  /* Chip select pin */
#define LCD_NUM_RST GPIO_NUM_38 /* Reset pin */

/* IO operations */
#define LCD_WR(x)                                                              \
    do                                                                         \
    {                                                                          \
        x ? (gpio_set_level(LCD_NUM_WR, 1)) : (gpio_set_level(LCD_NUM_WR, 0)); \
    } while (0)

#define LCD_CS(x)                                                              \
    do                                                                         \
    {                                                                          \
        x ? (gpio_set_level(LCD_NUM_CS, 1)) : (gpio_set_level(LCD_NUM_CS, 0)); \
    } while (0)

#define LCD_PWR(x)                                                             \
    do                                                                         \
    {                                                                          \
        x ? (gpio_set_level(LCD_NUM_BL, 1)) : (gpio_set_level(LCD_NUM_BL, 0)); \
    } while (0)

#define LCD_RST(x)                                                               \
    do                                                                           \
    {                                                                            \
        x ? (gpio_set_level(LCD_NUM_RST, 1)) : (gpio_set_level(LCD_NUM_RST, 0)); \
    } while (0)

/* Common color values */
#define WHITE 0xFFFF   /* White */
#define BLACK 0x0000   /* Black */
#define RED 0xF800     /* Red */
#define GREEN 0x07E0   /* Green */
#define BLUE 0x001F    /* Blue */
#define MAGENTA 0XF81F /* Magenta/Purple = BLUE + RED */
#define YELLOW 0XFFE0  /* Yellow = GREEN + RED */
#define CYAN 0X07FF    /* Cyan = GREEN + BLUE */

/* Less common color values */
#define BROWN 0XBC40      /* Brown */
#define BRRED 0XFC07      /* Brown-red */
#define GRAY 0X8430       /* Gray */
#define DARKBLUE 0X01CF   /* Dark blue */
#define LIGHTBLUE 0X7D7C  /* Light blue */
#define GRAYBLUE 0X5458   /* Gray-blue */
#define LIGHTGREEN 0X841F /* Light green */
#define LGRAY 0XC618      /* Light gray (panel background) */
#define LGRAYBLUE 0XA651  /* Light gray-blue (middle layer color) */
#define LBBLUE 0X2B12     /* Light brown-blue (selection bar highlight color) */

/* Scan direction definitions */
#define L2R_U2D 0 /* Left to right, top to bottom */
#define L2R_D2U 1 /* Left to right, bottom to top */
#define R2L_U2D 2 /* Right to left, top to bottom */
#define R2L_D2U 3 /* Right to left, bottom to top */
#define U2D_L2R 4 /* Top to bottom, left to right */
#define U2D_R2L 5 /* Top to bottom, right to left */
#define D2U_L2R 6 /* Bottom to top, left to right */
#define D2U_R2L 7 /* Bottom to top, right to left */

#define DFT_SCAN_DIR L2R_U2D /* Default scan direction */

    /* LCD information structure */
    typedef struct _lcd_obj_t
    {
        uint16_t width;   /* Width */
        uint16_t height;  /* Height */
        uint8_t dir;      /* Screen orientation: 0 = vertical, 1 = horizontal */
        uint16_t wramcmd; /* Write GRAM command */
        uint16_t setxcmd; /* Set X coordinate command */
        uint16_t setycmd; /* Set Y coordinate command */
        uint16_t wr;      /* Command/Data IO */
        uint16_t cs;      /* Chip select IO */
        uint16_t bl;      /* Backlight */
        uint16_t rst;     /* Reset */
    } lcd_obj_t;

/* LCD buffer size settings - modify with caution!
   Changing these values may affect lcd_clear/lcd_fill/lcd_draw_line functions */
#define LCD_TOTAL_BUF_SIZE (160 * 80 * 2) /* Total buffer size */
#define LCD_BUF_SIZE 2560                 /* Partial buffer size */

    /* Exported variables */
    extern lcd_obj_t lcd_self;
    extern uint8_t lcd_buf[LCD_TOTAL_BUF_SIZE];

    /* Function declarations */
    void lcd_init(void);                                                                                                  /* Initialize the LCD */
    void lcd_clear(uint16_t color);                                                                                       /* Clear the screen */
    void lcd_scan_dir(uint8_t dir);                                                                                       /* Set the LCD auto scan direction */
    void lcd_write_data(const uint8_t *data, int len);                                                                    /* Write data to the LCD */
    void lcd_write_data16(uint16_t data);                                                                                 /* Write 16-bit data to the LCD */
    void lcd_set_cursor(uint16_t xpos, uint16_t ypos);                                                                    /* Set the cursor position */
    void lcd_set_window(uint16_t xstar, uint16_t ystar, uint16_t xend, uint16_t yend);                                    /* Set the display window size */
    void lcd_fill(uint16_t sx, uint16_t sy, uint16_t ex, uint16_t ey, uint16_t color);                                    /* Fill a specific area with a single color */
    void lcd_show_num(uint16_t x, uint16_t y, uint32_t num, uint8_t len, uint8_t size, uint16_t color);                   /* Display a number with len digits */
    void lcd_show_xnum(uint16_t x, uint16_t y, uint32_t num, uint8_t len, uint8_t size, uint8_t mode, uint16_t color);    /* Display an extended number with len digits */
    void lcd_show_string(uint16_t x, uint16_t y, uint16_t width, uint16_t height, uint8_t size, char *p, uint16_t color); /* Display a string */
    void lcd_draw_rectangle(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, uint16_t color);                          /* Draw a rectangle */
    void lcd_draw_hline(uint16_t x, uint16_t y, uint16_t len, uint16_t color);                                            /* Draw a horizontal line */
    void lcd_draw_line(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t color);                               /* Draw a straight or slanted line */
    void lcd_draw_pixel(uint16_t x, uint16_t y, uint16_t color);                                                          /* Draw a pixel */
    void lcd_show_char(uint16_t x, uint16_t y, uint8_t chr, uint8_t size, uint8_t mode, uint16_t color);                  /* Display a character at a specified position */

#ifdef __cplusplus
}
#endif