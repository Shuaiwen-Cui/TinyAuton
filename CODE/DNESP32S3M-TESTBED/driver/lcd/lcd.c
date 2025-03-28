/**
 * @file lcd.c
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief  LCD driver source file
 * @version 1.0
 * @date 2024-11-18
 * @ref Alientek LCD Driver
 * @copyright Copyright (c) 2024
 *
 */

#include "lcd.h"
#include "lcdfont.h"

spi_device_handle_t MY_LCD_Handle;
uint8_t lcd_buf[LCD_TOTAL_BUF_SIZE];
lcd_obj_t lcd_self;

/* The LCD requires a set of initialization commands/parameters.
   They are stored in this structure. */
typedef struct
{
    uint8_t cmd;       /* Command byte */
    uint8_t data[16];  /* Data array for the command */
    uint8_t databytes; /* Number of data bytes; Bit 7 = post-command delay; 0xFF = end of commands */
} lcd_init_cmd_t;

/**
 * @brief       Sends a command to the LCD using polling mode.
 *              Blocks until the transmission is complete.
 *              (Since the data transmission is minimal, polling mode improves speed.
 *              The overhead of interrupt handling exceeds that of polling in this case.)
 * @param       cmd 8-bit command data to be transmitted
 * @retval      None
 */
void lcd_write_cmd(const uint8_t cmd)
{
    LCD_WR(0);                          // Set write signal to 0
    spi2_write_cmd(MY_LCD_Handle, cmd); // Transmit the command via SPI
}

/**
 * @brief       Sends data to the LCD using polling mode.
 *              Blocks until the transmission is complete.
 *              (Since the data transmission is minimal, polling mode improves speed.
 *              The overhead of interrupt handling exceeds that of polling in this case.)
 * @param       data Pointer to the 8-bit data to be transmitted
 * @param       len  Length of the data to be transmitted
 * @retval      None
 */
void lcd_write_data(const uint8_t *data, int len)
{
    LCD_WR(1);                                 // Set write signal to 1
    spi2_write_data(MY_LCD_Handle, data, len); // Transmit the data via SPI
}

/**
 * @brief       Sends 16-bit data to the LCD using polling mode.
 *              Blocks until the transmission is complete.
 *              (Since the data transmission is minimal, polling mode improves speed.
 *              The overhead of interrupt handling exceeds that of polling in this case.)
 * @param       data 16-bit data to be transmitted
 * @retval      None
 */
void lcd_write_data16(uint16_t data)
{
    uint8_t dataBuf[2] = {0, 0};
    dataBuf[0] = data >> 8;                     // Extract the high byte
    dataBuf[1] = data & 0xFF;                   // Extract the low byte
    LCD_WR(1);                                  // Set write signal to 1
    spi2_write_data(MY_LCD_Handle, dataBuf, 2); // Transmit the 16-bit data via SPI
}

/**
 * @brief       Set the window size
 * @param       xstar: Top-left corner x-axis coordinate
 * @param       ystar: Top-left corner y-axis coordinate
 * @param       xend: Bottom-right corner x-axis coordinate
 * @param       yend: Bottom-right corner y-axis coordinate
 * @retval      None
 */
void lcd_set_window(uint16_t xstar, uint16_t ystar, uint16_t xend, uint16_t yend)
{
    uint8_t databuf[4] = {0, 0, 0, 0};

    if (lcd_self.dir == 1) /* Landscape orientation */
    {
        databuf[0] = (xstar + 1) >> 8;
        databuf[1] = 0xFF & (xstar + 1);
        databuf[2] = (xend + 1) >> 8;
        databuf[3] = 0xFF & (xend + 1);
        lcd_write_cmd(lcd_self.setxcmd);
        lcd_write_data(databuf, 4);

        databuf[0] = (ystar + 26) >> 8;
        databuf[1] = 0xFF & (ystar + 26);
        databuf[2] = (yend + 26) >> 8;
        databuf[3] = 0xFF & (yend + 26);
        lcd_write_cmd(lcd_self.setycmd);
        lcd_write_data(databuf, 4);
    }
    else /* Portrait orientation */
    {
        databuf[0] = (xstar + 26) >> 8;
        databuf[1] = 0xFF & (xstar + 26);
        databuf[2] = (xend + 26) >> 8;
        databuf[3] = 0xFF & (xend + 26);
        lcd_write_cmd(lcd_self.setxcmd);
        lcd_write_data(databuf, 4);

        databuf[0] = (ystar + 1) >> 8;
        databuf[1] = 0xFF & (ystar + 1);
        databuf[2] = (yend + 26) >> 8;
        databuf[3] = 0xFF & (yend + 26);
        lcd_write_cmd(lcd_self.setycmd);
        lcd_write_data(databuf, 4);
    }

    lcd_write_cmd(lcd_self.wramcmd); /* Begin writing to GRAM */
}

/**
 * @brief       Clear the LCD screen with a specified color
 * @param       color The color to use for clearing the screen
 * @retval      None
 */
void lcd_clear(uint16_t color)
{
    uint16_t i, j;
    uint8_t data[2] = {0};

    data[0] = color >> 8; // High byte of the color
    data[1] = color;      // Low byte of the color

    lcd_set_window(0, 0, lcd_self.width - 1, lcd_self.height - 1); // Set the full screen as the window

    for (j = 0; j < LCD_BUF_SIZE / 2; j++) // Prepare buffer with the color data
    {
        lcd_buf[j * 2] = data[0];
        lcd_buf[j * 2 + 1] = data[1];
    }

    for (i = 0; i < (LCD_TOTAL_BUF_SIZE / LCD_BUF_SIZE); i++) // Write the buffer repeatedly to cover the entire screen
    {
        lcd_write_data(lcd_buf, LCD_BUF_SIZE);
    }
}

/**
 * @brief       Fill a specified area with a single color
 * @param       (sx, sy), (ex, ey): Coordinates of the opposite corners of the rectangle to fill.
 *              The region size is: (ex - sx + 1) * (ey - sy + 1).
 * @param       color: The color to fill the area with (16-bit color for compatibility with LTDC)
 * @retval      None
 */
void lcd_fill(uint16_t sx, uint16_t sy, uint16_t ex, uint16_t ey, uint16_t color)
{
    uint16_t i;
    uint16_t j;
    uint16_t width;
    uint16_t height;

    width = ex - sx + 1;            // Calculate the width of the fill area
    height = ey - sy + 1;           // Calculate the height of the fill area
    lcd_set_window(sx, sy, ex, ey); // Set the fill area as the active window

    for (i = 0; i < height; i++) // Iterate over the height of the area
    {
        for (j = 0; j < width; j++) // Iterate over the width of the area
        {
            lcd_write_data16(color); // Write the color data to the LCD
        }
    }
    lcd_set_window(sx, sy, ex, ey); // Reset the window to the original fill area
}

/**
 * @brief       Set the position of the cursor
 * @param       xpos: X-axis coordinate of the top-left corner
 * @param       ypos: Y-axis coordinate of the top-left corner
 * @retval      None
 */
void lcd_set_cursor(uint16_t xpos, uint16_t ypos)
{
    lcd_set_window(xpos, ypos, xpos, ypos); // Set the window to a single pixel at the cursor position
}

/**
 * @brief       Set the auto-scan direction of the LCD (not applicable for RGB screens)
 * @param       dir: 0~7, representing 8 directions (specific definitions can be found in lcd.h)
 * @retval      None
 */
void lcd_scan_dir(uint8_t dir)
{
    uint8_t regval = 0;
    uint8_t dirreg = 0;
    uint16_t temp;

    /* For landscape mode, the 1963 IC does not change scan direction, other ICs do.
       For portrait mode, the 1963 IC changes direction, other ICs do not. */
    if (lcd_self.dir == 1)
    {
        dir = 1;
    }
    else
    {
        dir = 0;
    }

    switch (dir)
    {
    case 0: /* Portrait mode, top-right corner is (0, 0) */
        regval |= 0x08;
        break;

    case 1: /* Landscape mode, top-left corner is (0, 0) */
        regval |= 0xA8;
        break;

    case 2: /* Portrait mode, bottom-left corner is (0, 0) */
        regval |= 0xC8;
        break;

    case 3: /* Landscape mode, bottom-right corner is (0, 0) */
        regval |= 0x78;
        break;
    }

    dirreg = 0x36; /* For most driver ICs, the 0x36 register controls this setting */

    uint8_t data_send[1] = {regval}; // Data to be sent to the register

    lcd_write_cmd(dirreg);        // Write command to set direction
    lcd_write_data(data_send, 1); // Send the direction data

    if (regval & 0x20)
    {
        if (lcd_self.width < lcd_self.height) /* Swap X and Y dimensions */
        {
            temp = lcd_self.width;
            lcd_self.width = lcd_self.height;
            lcd_self.height = temp;
        }
    }
    else
    {
        if (lcd_self.width > lcd_self.height) /* Swap X and Y dimensions */
        {
            temp = lcd_self.width;
            lcd_self.width = lcd_self.height;
            lcd_self.height = temp;
        }
    }

    lcd_set_window(0, 0, lcd_self.width, lcd_self.height); // Reset the window size
}

/**
 * @brief       Set the display orientation of the LCD
 * @param       dir: 0 for portrait mode; 1 for landscape mode
 * @retval      None
 */
void lcd_display_dir(uint8_t dir)
{
    lcd_self.dir = dir;

    if (lcd_self.dir == 0) /* Portrait mode */
    {
        lcd_self.width = 80;     // Default width for portrait
        lcd_self.height = 160;   // Default height for portrait
        lcd_self.wramcmd = 0x2C; // Write RAM command
        lcd_self.setxcmd = 0x2A; // Set X coordinate command
        lcd_self.setycmd = 0x2B; // Set Y coordinate command
    }
    else /* Landscape mode */
    {
        lcd_self.width = 160;    // Default width for landscape
        lcd_self.height = 80;    // Default height for landscape
        lcd_self.wramcmd = 0x2C; // Write RAM command
        lcd_self.setxcmd = 0x2A; // Set X coordinate command
        lcd_self.setycmd = 0x2B; // Set Y coordinate command
    }

    lcd_scan_dir(lcd_self.dir); // Set default scan direction
}

/**
 * @brief       Perform a hardware reset on the LCD
 * @param       None
 * @retval      None
 */
void lcd_hard_reset(void)
{
    /* Reset the display */
    LCD_RST(0);      // Pull the reset pin low
    vTaskDelay(200); // Delay for 200ms
    LCD_RST(1);      // Pull the reset pin high
    vTaskDelay(200); // Delay for 200ms
}

/**
 * @brief       Draw a single pixel on the LCD
 * @param       x: X-axis coordinate of the pixel
 * @param       y: Y-axis coordinate of the pixel
 * @param       color: Color value of the pixel
 * @retval      None
 */
void lcd_draw_pixel(uint16_t x, uint16_t y, uint16_t color)
{
    lcd_set_cursor(x, y);    // Set the cursor to the specified coordinates
    lcd_write_data16(color); // Write the color data for the pixel
}

/**
 * @brief       Draw a line (straight or diagonal) on the LCD
 * @param       x1, y1: Starting point coordinates
 * @param       x2, y2: Ending point coordinates
 * @param       color: Color value to fill the line
 * @retval      None
 */
void lcd_draw_line(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t color)
{
    uint16_t t;
    int xerr = 0, yerr = 0, delta_x, delta_y, distance;
    int incx, incy, urow, ucol;

    delta_x = x2 - x1; /* Calculate the x-coordinate increment */
    delta_y = y2 - y1; /* Calculate the y-coordinate increment */
    urow = x1;         /* Initialize the current x-coordinate */
    ucol = y1;         /* Initialize the current y-coordinate */

    if (delta_x > 0)
    {
        incx = 1; /* Set step direction for x */
    }
    else if (delta_x == 0)
    {
        incx = 0; /* Vertical line */
    }
    else
    {
        incx = -1;
        delta_x = -delta_x;
    }
    if (delta_y > 0)
    {
        incy = 1; /* Set step direction for y */
    }
    else if (delta_y == 0)
    {
        incy = 0; /* Horizontal line */
    }
    else
    {
        incy = -1;
        delta_y = -delta_y;
    }

    if (delta_x > delta_y)
    {
        distance = delta_x; /* Use x as the primary axis */
    }
    else
    {
        distance = delta_y; /* Use y as the primary axis */
    }

    for (t = 0; t <= distance + 1; t++) /* Draw the line */
    {
        lcd_draw_pixel(urow, ucol, color); /* Plot the pixel */
        xerr += delta_x;
        yerr += delta_y;

        if (xerr > distance)
        {
            xerr -= distance;
            urow += incx; /* Increment x */
        }

        if (yerr > distance)
        {
            yerr -= distance;
            ucol += incy; /* Increment y */
        }
    }
}

/**
 * @brief       Draw a horizontal line
 * @param       x: Starting x-coordinate
 * @param       y: Starting y-coordinate
 * @param       len: Length of the line
 * @param       color: Color of the line
 * @retval      None
 */
void lcd_draw_hline(uint16_t x, uint16_t y, uint16_t len, uint16_t color)
{
    if ((len == 0) || (x > lcd_self.width) || (y > lcd_self.height))
        return;

    lcd_fill(x, y, x + len - 1, y, color); // Fill a rectangle with height 1 as the horizontal line
}

/**
 * @brief       Draw a rectangle
 * @param       x0, y0: Coordinates of the top-left corner
 * @param       x1, y1: Coordinates of the bottom-right corner
 * @param       color: Color of the rectangle
 * @retval      None
 */
void lcd_draw_rectangle(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, uint16_t color)
{
    lcd_draw_line(x0, y0, x1, y0, color); // Top edge
    lcd_draw_line(x0, y0, x0, y1, color); // Left edge
    lcd_draw_line(x0, y1, x1, y1, color); // Bottom edge
    lcd_draw_line(x1, y0, x1, y1, color); // Right edge
}

/**
 * @brief       Draw a circle
 * @param       x0, y0: Coordinates of the circle's center
 * @param       r: Radius of the circle
 * @param       color: Color of the circle
 * @retval      None
 */
void lcd_draw_circle(uint16_t x0, uint16_t y0, uint16_t r, uint16_t color)
{
    int a, b;
    int di;

    a = 0;             // X-coordinate offset
    b = r;             // Y-coordinate offset
    di = 3 - (r << 1); // Decision parameter for midpoint algorithm

    while (a <= b)
    {
        lcd_draw_pixel(x0 - b, y0 - a, color); // 8-way symmetry
        lcd_draw_pixel(x0 + b, y0 - a, color);
        lcd_draw_pixel(x0 - a, y0 + b, color);
        lcd_draw_pixel(x0 - b, y0 - a, color);
        lcd_draw_pixel(x0 - a, y0 - b, color);
        lcd_draw_pixel(x0 + b, y0 + a, color);
        lcd_draw_pixel(x0 + a, y0 - b, color);
        lcd_draw_pixel(x0 + a, y0 + b, color);
        lcd_draw_pixel(x0 - b, y0 + a, color);
        a++;

        if (di < 0) // Update decision parameter
        {
            di += 4 * a + 6;
        }
        else
        {
            di += 10 + 4 * (a - b);
            b--;
        }

        lcd_draw_pixel(x0 + a, y0 + b, color); // Draw additional point due to symmetry
    }
}

/**
 * @brief       Display a character at a specified position
 * @param       x, y  : Coordinates where the character will be displayed
 * @param       chr   : Character to display, from " " to "~"
 * @param       size  : Font size, options: 12/16/24/32
 * @param       mode  : Overlay mode (1) or non-overlay mode (0)
 * @param       color : Color of the character
 * @retval      None
 */
void lcd_show_char(uint16_t x, uint16_t y, uint8_t chr, uint8_t size, uint8_t mode, uint16_t color)
{
    uint8_t temp = 0, t1 = 0, t = 0;
    uint8_t *pfont = 0;
    uint8_t csize = 0; /* Number of bytes for the font matrix of one character */
    uint16_t colortemp = 0;
    uint8_t sta = 0;

    csize = (size / 8 + ((size % 8) ? 1 : 0)) * (size / 2); /* Calculate font matrix byte size for one character */
    chr = chr - ' ';                                        /* Offset value (ASCII font starts at space character) */

    if ((x > (lcd_self.width - size / 2)) || (y > (lcd_self.height - size)))
    {
        return;
    }

    lcd_set_window(x, y, x + size / 2 - 1, y + size - 1); /* Set the window area for the character */

    switch (size)
    {
    case 12:
        pfont = (uint8_t *)asc2_1206[chr]; /* Use 12x6 font */
        break;

    case 16:
        pfont = (uint8_t *)asc2_1608[chr]; /* Use 16x8 font */
        break;

    case 24:
        pfont = (uint8_t *)asc2_2412[chr]; /* Use 24x12 font */
        break;

    case 32:
        pfont = (uint8_t *)asc2_3216[chr]; /* Use 32x16 font */
        break;

    default:
        return;
    }

    if (size != 24)
    {
        csize = (size / 8 + ((size % 8) ? 1 : 0)) * (size / 2);

        for (t = 0; t < csize; t++)
        {
            temp = pfont[t]; /* Retrieve font matrix data */

            for (t1 = 0; t1 < 8; t1++)
            {
                if (temp & 0x80)
                {
                    colortemp = color;
                }
                else if (mode == 0) /* Transparent point, do not display */
                {
                    colortemp = 0xFFFF;
                }

                lcd_write_data16(colortemp);
                temp <<= 1;
            }
        }
    }
    else
    {
        csize = (size * 16) / 8;

        for (t = 0; t < csize; t++)
        {
            temp = asc2_2412[chr][t];

            if (t % 2 == 0)
            {
                sta = 8;
            }
            else
            {
                sta = 4;
            }

            for (t1 = 0; t1 < sta; t1++)
            {
                if (temp & 0x80)
                {
                    colortemp = color;
                }
                else if (mode == 0) /* Transparent point, do not display */
                {
                    colortemp = 0xFFFF;
                }

                lcd_write_data16(colortemp);
                temp <<= 1;
            }
        }
    }
}

/**
 * @brief       Function to calculate m raised to the power of n (m^n)
 * @param       m: Base value
 * @param       n: Exponent value
 * @retval      Result of m raised to the power of n (m^n)
 */
uint32_t lcd_pow(uint8_t m, uint8_t n)
{
    uint32_t result = 1;

    while (n--)
        result *= m; // Multiply the result by m, n times

    return result;
}

/**
 * @brief       Display a number with a specified length
 * @param       x, y : Starting coordinates
 * @param       num  : Number to display (0 ~ 2^32)
 * @param       len  : Number of digits to display
 * @param       size : Font size (12/16/24/32)
 * @param       color: Color of the number
 * @retval      None
 */
void lcd_show_num(uint16_t x, uint16_t y, uint32_t num, uint8_t len, uint8_t size, uint16_t color)
{
    uint8_t t, temp;
    uint8_t enshow = 0;

    for (t = 0; t < len; t++) /* Loop through the number of digits to display */
    {
        temp = (num / lcd_pow(10, len - t - 1)) % 10; /* Extract the digit at the current position */

        if (enshow == 0 && t < (len - 1)) /* If display not enabled and there are more digits */
        {
            if (temp == 0)
            {
                lcd_show_char(x + (size / 2) * t, y, ' ', size, 0, color); /* Display a space as a placeholder */
                continue;                                                  /* Skip to the next digit */
            }
            else
            {
                enshow = 1; /* Enable display */
            }
        }

        lcd_show_char(x + (size / 2) * t, y, temp + '0', size, 0, color); /* Display the digit as a character */
    }
}

/**
 * @brief       Extended display of a number with a specified length (including leading zeros)
 * @param       x, y : Starting coordinates
 * @param       num  : Number to display (0 ~ 2^32)
 * @param       len  : Number of digits to display
 * @param       size : Font size (12/16/24/32)
 * @param       mode : Display mode
 *              [7]: 0, no padding; 1, pad with zeros.
 *              [6:1]: Reserved
 *              [0]: 0, non-overlay display; 1, overlay display.
 * @param       color: Color of the number
 * @retval      None
 */
void lcd_show_xnum(uint16_t x, uint16_t y, uint32_t num, uint8_t len, uint8_t size, uint8_t mode, uint16_t color)
{
    uint8_t t, temp;
    uint8_t enshow = 0;

    for (t = 0; t < len; t++) /* Loop through the number of digits to display */
    {
        temp = (num / lcd_pow(10, len - t - 1)) % 10; /* Extract the digit at the current position */

        if (enshow == 0 && t < (len - 1)) /* If display not enabled and there are more digits */
        {
            if (temp == 0)
            {
                if (mode & 0x80) /* Pad with zero if enabled */
                {
                    lcd_show_char(x + (size / 2) * t, y, '0', size, mode & 0x01, color); /* Display '0' as placeholder */
                }
                else
                {
                    lcd_show_char(x + (size / 2) * t, y, ' ', size, mode & 0x01, color); /* Display space as placeholder */
                }
                continue;
            }
            else
            {
                enshow = 1; /* Enable display */
            }
        }
        lcd_show_char(x + (size / 2) * t, y, temp + '0', size, mode & 0x01, color); /* Display the digit as a character */
    }
}

/**
 * @brief       Display a string within a specified area
 * @param       x, y         : Starting coordinates
 * @param       width, height: Dimensions of the display area
 * @param       size         : Font size (12/16/24/32)
 * @param       p            : Pointer to the string
 * @param       color        : Color of the string
 * @retval      None
 */
void lcd_show_string(uint16_t x, uint16_t y, uint16_t width, uint16_t height, uint8_t size, char *p, uint16_t color)
{
    uint8_t x0 = x; // Save the initial x-coordinate
    width += x;     // Calculate the maximum x-boundary
    height += y;    // Calculate the maximum y-boundary

    while ((*p <= '~') && (*p >= ' ')) /* Check if the character is valid */
    {
        if (x >= width) // If the x-coordinate exceeds the width
        {
            x = x0;    // Reset to the initial x-coordinate
            y += size; // Move to the next line
        }

        if (y >= height) // If the y-coordinate exceeds the height, exit
            break;

        lcd_show_char(x, y, *p, size, 0, color); // Display the current character
        x += size / 2;                           // Move to the next character position
        p++;                                     // Advance the string pointer
    }
}

/**
 * @brief       Turn on the LCD
 * @param       None
 * @retval      None (Initialization successful)
 */
void lcd_on(void)
{
    LCD_PWR(1);     // Power on the LCD
    vTaskDelay(10); // Delay for 10 ms
}

/**
 * @brief       Turn off the LCD
 * @param       None
 * @retval      None (Initialization successful)
 */
void lcd_off(void)
{
    LCD_PWR(0);     // Power off the LCD
    vTaskDelay(10); // Delay for 10 ms
}

/**
 * @brief       Initialize the LCD
 * @param       None
 * @retval      None
 */
void lcd_init(void)
{
    int cmd = 0;
    esp_err_t ret = 0;

    lcd_self.dir = 0;
    lcd_self.wr = LCD_NUM_WR;   /* Configure WR pin */
    lcd_self.cs = LCD_NUM_CS;   /* Configure CS pin */
    lcd_self.bl = LCD_NUM_BL;   /* Configure BL pin */
    lcd_self.rst = LCD_NUM_RST; /* Configure RST pin */

    gpio_config_t gpio_init_struct;

    /* SPI driver interface configuration */
    spi_device_interface_config_t devcfg = {
        .clock_speed_hz = 60 * 1000 * 1000, /* SPI clock frequency */
        .mode = 0,                          /* SPI mode 0 */
        .spics_io_num = lcd_self.cs,        /* SPI device pin */
        .queue_size = 7,                    /* Transaction queue size */
    };

    /* Add SPI device to the bus */
    ret = spi_bus_add_device(SPI2_HOST, &devcfg, &MY_LCD_Handle); /* Configure SPI bus device */
    ESP_ERROR_CHECK(ret);

    /* Configure WR pin */
    gpio_init_struct.intr_type = GPIO_INTR_DISABLE;        /* Disable pin interrupt */
    gpio_init_struct.mode = GPIO_MODE_OUTPUT;              /* Set as output mode */
    gpio_init_struct.pin_bit_mask = 1ull << lcd_self.wr;   /* Configure pin bit mask */
    gpio_init_struct.pull_down_en = GPIO_PULLDOWN_DISABLE; /* Disable pull-down */
    gpio_init_struct.pull_up_en = GPIO_PULLUP_ENABLE;      /* Enable pull-up */
    gpio_config(&gpio_init_struct);                        /* Apply pin configuration */
    /* Configure BL pin */
    gpio_init_struct.pin_bit_mask = 1ull << lcd_self.bl;  /* Configure pin bit mask */
    gpio_init_struct.pull_down_en = GPIO_PULLDOWN_ENABLE; /* Enable pull-down */
    gpio_init_struct.pull_up_en = GPIO_PULLUP_DISABLE;    /* Disable pull-up */
    gpio_config(&gpio_init_struct);                       /* Apply pin configuration */
    /* Configure RST pin */
    gpio_init_struct.pin_bit_mask = 1ull << lcd_self.rst;  /* Configure pin bit mask */
    gpio_init_struct.pull_down_en = GPIO_PULLDOWN_DISABLE; /* Disable pull-down */
    gpio_init_struct.pull_up_en = GPIO_PULLUP_ENABLE;      /* Enable pull-up */
    gpio_config(&gpio_init_struct);                        /* Apply pin configuration */

    lcd_off();        /* Turn off the LCD */
    lcd_hard_reset(); /* Perform hardware reset */

    /* Initialization sequence for 0.96 inch LCD */
    lcd_init_cmd_t ili_init_cmds[] =
        {
            {0x11, {0}, 0x80},
            {0x21, {0}, 0x80},
            {0xB1, {0x05, 0x3A, 0x3A}, 3},
            {0xB2, {0x05, 0x3A, 0x3A}, 3},
            {0xB3, {0x05, 0x3A, 0x3A, 0x05, 0x3A, 0x3A}, 6},
            {0xB4, {0x03}, 1},
            {0xC0, {0x62, 0x02, 0x04}, 3},
            {0xC1, {0xC0}, 1},
            {0xC2, {0x0D, 0x00}, 2},
            {0xC3, {0x8D, 0x6A}, 2},
            {0xC4, {0x8D, 0xEE}, 2},
            {0xC5, {0x0E}, 1},
            {0xE0, {0x10, 0x0E, 0x02, 0x03, 0x0E, 0x07, 0x02, 0x07, 0x0A, 0x12, 0x27, 0x37, 0x00, 0x0D, 0x0E, 0x10}, 16},
            {0xE1, {0x10, 0x0E, 0x03, 0x03, 0x0F, 0x06, 0x02, 0x08, 0x0A, 0x13, 0x26, 0x36, 0x00, 0x0D, 0x0E, 0x10}, 16},
            {0x3A, {0x05}, 1},
            {0x36, {0xA8}, 1},
            {0x29, {0}, 0x80},
            {0, {0}, 0xFF},
        };

    /* Send initialization sequence */
    while (ili_init_cmds[cmd].databytes != 0xFF)
    {
        lcd_write_cmd(ili_init_cmds[cmd].cmd);
        lcd_write_data(ili_init_cmds[cmd].data, ili_init_cmds[cmd].databytes & 0x1F);

        if (ili_init_cmds[cmd].databytes & 0x80)
        {
            vTaskDelay(120);
        }

        cmd++;
    }

    lcd_display_dir(1); /* Set screen orientation */
    lcd_clear(WHITE);   /* Clear the screen */
    lcd_on();           /* Turn on the LCD */
}