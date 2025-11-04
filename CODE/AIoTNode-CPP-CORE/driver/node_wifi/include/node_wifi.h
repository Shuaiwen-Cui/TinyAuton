/**
 * @file node_wifi.h
 * @author SHUAIWEN CUI (SHUAIWEN001@e.ntu.edu.sg)
 * @brief This file contains the function prototypes for wifi connection using WPA2 enterprise.
 * @version 1.0
 * @date 2025-02-24
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

/* Dependencies */
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_eap_client.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_netif.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Macros */
#define ENTERPRISE_WIFI_SSID "NTUSECURE"                    // SSID of WiFi
#define ENTERPRISE_WIFI_USERNAME "SHUAIWEN001@e.ntu.edu.sg" // Username
#define ENTERPRISE_WIFI_PASSWORD "Csw19950918%"             // Password

    /* Variables */
    extern const char *TAG_WIFI; // tag for logging
    extern EventGroupHandle_t wifi_event_group;
    extern const int CONNECTED_BIT;

    /* Function Prototypes */
    /**
     * @name wifi_sta_wpa2_init
     * @brief Initialize the WIFI station to connect to a WPA2 Enterprise network
     * @param void
     * @return esp_err_t
     */
    esp_err_t wifi_sta_wpa2_init(void);

#ifdef __cplusplus
}
#endif
