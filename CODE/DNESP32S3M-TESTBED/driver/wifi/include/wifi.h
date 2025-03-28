// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "freertos/event_groups.h"
// #include "esp_system.h"
// #include "esp_wifi.h"
// #include "esp_event.h"
// #include <netdb.h>
// #include "led.h"
// #include "lcd.h"

// #define DEFAULT_SSID        "CSW@CEE"
// #define DEFAULT_PWD         "88888888"
// #define WIFI_CONNECTED_BIT  BIT0
// #define WIFI_FAIL_BIT       BIT1

// /* WIFI默认配置 */
// #define WIFICONFIG()   {                            \
//     .sta = {                                        \
//         .ssid = DEFAULT_SSID,                       \
//         .password = DEFAULT_PWD,                    \
//         .threshold.authmode = WIFI_AUTH_WPA2_PSK,   \
//     },                                              \
// }

// /**
//  * @brief       WIFI初始化
//  * @param       无
//  * @retval      无
//  */
// void wifi_sta_init(void);